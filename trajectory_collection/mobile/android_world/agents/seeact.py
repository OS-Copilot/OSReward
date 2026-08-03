# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Screenshot-only tool-call agent for Android trajectory collection."""

from typing import Any
from collections import deque
import json
import re
import time

import numpy as np
from openai import OpenAI

from android_world.agents import base_agent
from android_world.agents import model_profiles
from android_world.env import interface
from android_world.env import json_action
from android_world.api.llm_api_utils import get_model_response

def _reverse_direction(direction: str | None) -> str:
  return {
      "up": "down",
      "down": "up",
      "left": "right",
      "right": "left",
  }.get(direction, "")


def _dir_from_coords(x0: float, y0: float, x1: float, y1: float) -> str:
  dx, dy = x1 - x0, y1 - y0
  if abs(dx) > abs(dy):
    return "right" if dx > 0 else "left"
  return "down" if dy > 0 else "up"


def qwen3vl_action_transform(
    action: str,
    arguments: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
  if action == "key":
    return {"action_type": "wait"}
  if action in ("click", "left_click"):
    coordinate = arguments.get("coordinate", [0, 0])
    x, y = coordinate
    return {"action_type": "click", "x": x / 1000 * width, "y": y / 1000 * height}
  if action == "long_press":
    coordinate = arguments.get("coordinate", [0, 0])
    x, y = coordinate
    return {
        "action_type": "long_press",
        "x": x / 1000 * width,
        "y": y / 1000 * height,
    }
  if action == "swipe":
    coordinate = arguments.get("coordinate", [0, 0])
    coordinate2 = arguments.get("coordinate2", [0, 0])
    x0, y0 = coordinate[0] / 1000 * width, coordinate[1] / 1000 * height
    x1, y1 = coordinate2[0] / 1000 * width, coordinate2[1] / 1000 * height
    direction = _dir_from_coords(x0, y0, x1, y1)
    return {"action_type": "scroll", "direction": _reverse_direction(direction)}
  if action == "type":
    return {"action_type": "input_text", "text": arguments.get("text", "")}
  if action == "system_button":
    button = arguments.get("button", "").lower()
    if button == "home":
      return {"action_type": "navigate_home"}
    if button == "back":
      return {"action_type": "navigate_back"}
    if button == "enter":
      return {"action_type": "keyboard_enter"}
    return {"action_type": "wait"}
  if action == "open":
    return {"action_type": "open_app", "app_name": arguments.get("text", "")}
  if action == "wait":
    return {"action_type": "wait"}
  if action == "answer":
    return {"action_type": "answer", "text": arguments.get("text", "")}
  if action == "terminate":
    status = arguments.get("status", "").lower()
    if status == "success":
      return {"action_type": "status", "goal_status": "complete"}
    if status == "failure":
      return {"action_type": "status", "goal_status": "infeasible"}
    return {"action_type": "status", "goal_status": "infeasible"}
  return {"action_type": "wait"}


def _parse_tool_call_json(block: str) -> dict[str, Any] | None:
  m = re.search(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", block)
  if not m:
    return None
  payload = m.group(1).strip()
  try:
    return json.loads(payload)
  except Exception:
    return None


class ToolCallAgent(base_agent.EnvironmentInteractingAgent):
  """Model-agnostic Android GUI agent driven by `<tool_call>` JSON output.

  The prompt format and history extraction come from a `ModelProfile`, resolved
  once from the model name (see `model_profiles.py`), so this class contains no
  per-model branching.
  """

  def __init__(
      self,
      env: interface.AsyncEnv,
      name: str = "ToolCallAgent",
      wait_after_action_seconds: float = 2.0,
      model_base_url: str = "http://127.0.0.1:8000/v1",
      model_api_key: str = "EMPTY",
      model_name: str = "",
      extra_headers: dict[str, str] | None = None,
      model_profile: str = "",
  ):
    super().__init__(env, name)
    self.wait_after_action_seconds = wait_after_action_seconds
    self.model_name = model_name
    # Raises on an unknown model rather than silently picking a prompt format.
    self.profile = model_profiles.resolve(model_name, model_profile)
    self.client = OpenAI(
        api_key=model_api_key,
        base_url=model_base_url,
        default_headers=extra_headers,
    )
    self.step_his: str = ""
    self.turn_number: int = 0
    self.last_action: str | None = None
    self.repeat_time: int = 0
    self.max_retry: int = 3
    self.temperature: int = 0
    self.timeout: int = 60

    # Keep the last N screenshots to provide multi-frame context.
    self.last_N: int = 3
    self._recent_screenshots: deque[np.ndarray] = deque(maxlen=self.last_N)

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.step_his = ""
    self.turn_number = 0
    self.last_action = None
    self.repeat_time = 0
    self._recent_screenshots.clear()

  @staticmethod
  def _to_base64_png(image: np.ndarray) -> str:
    import base64
    from io import BytesIO
    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.fromarray(image).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

  def step(self, instruction: str) -> base_agent.AgentInteractionResult:
    self.turn_number += 1

    state = self.get_post_transition_state()
    screenshot = state.pixels.copy()
    model_screenshot = screenshot[:, :, ::-1]
    height, width = model_screenshot.shape[:2]

    # Accumulate recent screenshots for multi-frame context.
    self._recent_screenshots.append(model_screenshot)

    system_prompt = self.profile.system_prompt
    user_prompt = self.profile.user_prompt.format(
        instruction=instruction, history=self.step_his
    )

    # Build user content with text + last N screenshots.
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for img in list(self._recent_screenshots):
      user_content.append(
          {"type": "image_url", "image_url": {"url": self._to_base64_png(img)}}
      )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    completion = get_model_response(
          model_url=str(self.client.base_url),
          model_name=self.model_name,
          model_token=self.client.api_key,
          messages=messages,
          tool_schemas=None,
          agent_logger=None,
          max_retry_num=self.max_retry,
          temperature=self.temperature,
          timeout=self.timeout
      )
    response = completion.content or ""

    print(response)
    print("=" * 50)

    args = None
    tool_call = _parse_tool_call_json(response)
    if not tool_call:
      return base_agent.AgentInteractionResult(
          True,
          {
              "summary": "No <tool_call> JSON found in model output.",
              "response": response,
              "screenshot": screenshot,
              "step_history": self.step_his,
              "parsed": None,
              "tool_call_args": None,
              "turn_number": self.turn_number,
              "screen_width": width,
              "screen_height": height,
          },
      )

    op_text = self.profile.extract_history(response)
    self.step_his += f"Step {self.turn_number}: {op_text}; "

    args = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else {}
    action_name = args.get("action", "")
    try:
      parsed = qwen3vl_action_transform(action_name, args, width, height)
      print(parsed)
    except Exception as e:
      return base_agent.AgentInteractionResult(
          True,
          {
              "summary": f"Failed to transform tool-call into action: {e}",
              "response": response,
              "tool_call": tool_call,
              "screenshot": screenshot,
              "step_history": self.step_his,
              "parsed": None,
              "tool_call_args": args,
              "turn_number": self.turn_number,
              "screen_width": width,
              "screen_height": height,
          },
      )

    try:
      action_sig = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except Exception:
      action_sig = str(args)
    if self.last_action == action_sig:
      self.repeat_time += 1
    else:
      self.repeat_time = 0
    self.last_action = action_sig

    try:
      act = json_action.JSONAction(**parsed)
      self.env.execute_action(act)
      time.sleep(self.wait_after_action_seconds)
    except Exception:
      print("Failed to execute action:", parsed)

    if parsed.get("action_type") == "status":
      return base_agent.AgentInteractionResult(
          True,
          {
              "response": response,
              "step_history": self.step_his,
              "parsed": parsed,
              "screenshot": screenshot,
              "summary": None,
              "tool_call_args": args,
              "turn_number": self.turn_number,
              "screen_width": width,
              "screen_height": height,
          },
      )

    if self.repeat_time >= 10:
      return base_agent.AgentInteractionResult(
          True,
          {
              "summary": "Terminated due to repeated identical actions.",
              "response": response,
              "step_history": self.step_his,
              "parsed": parsed,
              "repeat_time": self.repeat_time,
              "screenshot": screenshot,
              "tool_call_args": args,
              "turn_number": self.turn_number,
              "screen_width": width,
              "screen_height": height,
          },
      )

    return base_agent.AgentInteractionResult(
        False,
        {
            "response": response,
            "step_history": self.step_his,
            "parsed": parsed,
            "screenshot": screenshot,
            "summary": None,
            "tool_call_args": args,
            "turn_number": self.turn_number,
            "screen_width": width,
            "screen_height": height,
        },
    )


# Backwards-compatible alias. The agent was originally Qwen3-VL specific; it is
# now model-agnostic (see `model_profiles.py`), but existing imports and saved
# run metadata still refer to it by the old name.
Qwen3VL = ToolCallAgent
