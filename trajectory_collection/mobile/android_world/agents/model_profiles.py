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

"""Model profiles for the tool-call GUI agent.

The agent itself is model-agnostic: it sends screenshots and reads back a
`<tool_call>` JSON block. What differs between model families is (a) the prompt
pair and (b) how per-step history is recovered from the response text. A
`ModelProfile` bundles those together so the agent never branches on the model
name, and adding a model is a single entry in `MODEL_TO_PROFILE`.
"""

import dataclasses
import re
from typing import Callable

from android_world.agents import PROMPT


# ---------------------------------------------------------------- extractors
# These recover the short human-readable summary of a step that gets appended
# to the running history string. They never affect which action is executed --
# that always comes from the <tool_call> JSON.


def extract_action_text(block: str) -> str:
  """Reads the `Action:` line emitted by the Thought/Action prompt format.

  Not used by any entry in `PROFILES`. It pairs with the Thought/Action format
  of `PROMPT.QWEN3VL_SYSTEM_PROMPT_LASTN`, which is not currently wired up --
  kept so that prompt can be enabled without rewriting the extractor. Do not
  attach it to a profile whose system prompt asks for `<conclusion>`.
  """
  m = re.search(r"Action:\s*(.+?)(?:\n<tool_call>|$)", block, flags=re.S)
  if not m:
    return ""
  text = m.group(1).strip()
  if text.startswith('"') and text.endswith('"'):
    text = text[1:-1]
  return text.replace("\n", " ")


def extract_thinking_text(block: str) -> str:
  """Extracts content from <thinking>...</thinking> tags."""
  m = re.search(
      r"<thinking>\s*([\s\S]*?)\s*</thinking>", block or "", flags=re.I
  )
  return m.group(1).strip() if m else ""


def extract_conclusion_text(block: str) -> str:
  """Extracts content from <conclusion>...</conclusion> tags."""
  m = re.search(
      r"<conclusion>\s*([\s\S]*?)\s*</conclusion>", block or "", flags=re.I
  )
  return m.group(1).strip() if m else ""


def _conclusion_then_thinking(block: str) -> str:
  """Prefers <conclusion>, falling back to <thinking>."""
  return extract_conclusion_text(block) or extract_thinking_text(block)


# ------------------------------------------------------------------ profiles


@dataclasses.dataclass(frozen=True)
class ModelProfile:
  """Everything the agent needs to know about a model family.

  Attributes:
    profile_id: Short identifier, also accepted by `--model_profile`.
    system_prompt: System prompt for this family.
    user_prompt: User prompt template, formatted with `instruction` and
      `history`.
    extract_history: Pulls the step summary out of a raw model response.
  """

  profile_id: str
  system_prompt: str
  user_prompt: str
  extract_history: Callable[[str], str]


PROFILES: dict[str, ModelProfile] = {
    "qwen3vl": ModelProfile(
        profile_id="qwen3vl",
        system_prompt=PROMPT.QWEN3VL_SYSTEM_PROMPT,
        user_prompt=PROMPT.QWEN3VL_USER_PROMPT,
        extract_history=_conclusion_then_thinking,
    ),
    "gemini3": ModelProfile(
        profile_id="gemini3",
        system_prompt=PROMPT.GEMINI3PRO_SYSTEM_PROMPT,
        user_prompt=PROMPT.GEMINI3PRO_USER_PROMPT,
        extract_history=_conclusion_then_thinking,
    ),
}


# Model id (as sent to the API) -> profile. Add a line here to support a new
# model; add a PROFILES entry only if it needs a different prompt format.
MODEL_TO_PROFILE: dict[str, str] = {
    "qwen3-vl-235b-a22b-instruct": "qwen3vl",
    "gemini-3-pro-preview": "gemini3",
    "gemini-3-flash-preview": "gemini3",
    "gemini-3-flash": "gemini3",
}


def supported_models() -> list[str]:
  return sorted(MODEL_TO_PROFILE)


def supported_profiles() -> list[str]:
  return sorted(PROFILES)


def resolve(model_name: str, profile_override: str = "") -> ModelProfile:
  """Resolves the profile for a model.

  Args:
    model_name: Model id passed to the inference endpoint.
    profile_override: Force a profile for a model that is not in the table.
      Takes precedence over `model_name`.

  Returns:
    The matching `ModelProfile`.

  Raises:
    ValueError: If the override is unknown, or the model is not in the table
      and no override was given.
  """
  if profile_override:
    if profile_override not in PROFILES:
      raise ValueError(
          f"Unknown --model_profile {profile_override!r}. Expected one of:"
          f" {supported_profiles()}."
      )
    return PROFILES[profile_override]

  if not model_name:
    raise ValueError(
        "No model specified. Pass --model_name (or set MODEL_NAME). Supported"
        f" models: {supported_models()}."
    )

  if model_name not in MODEL_TO_PROFILE:
    raise ValueError(
        f"Unknown --model_name {model_name!r}. Supported models:"
        f" {supported_models()}. To evaluate a model that is not listed, pass"
        f" --model_profile with one of {supported_profiles()} to select the"
        " prompt format explicitly, or add an entry to MODEL_TO_PROFILE in"
        " android_world/agents/model_profiles.py."
    )

  return PROFILES[MODEL_TO_PROFILE[model_name]]
