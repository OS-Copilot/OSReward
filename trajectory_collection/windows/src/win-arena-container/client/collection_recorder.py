"""Recorder that aggregates multi-question trajectories into one JSON file."""
from __future__ import annotations

import base64
import datetime
import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

class CollectionRecorder:
    """Collect per-step screenshots + model outputs for multiple questions."""

    def __init__(
        self,
        output_dir: str,
        output_json: str,
        model: str,
        embed_base64: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = output_dir
        self.output_json = output_json
        self.embed_base64 = embed_base64
        self.screenshots_dir = os.path.join(output_dir, "screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)

        self.data: Dict[str, Any] = {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "metadata": metadata or {},
            "episodes": [],
        }
        self._current_episode: Optional[Dict[str, Any]] = None
        self._flush()

    def start_episode(
        self,
        question_id: str,
        instruction: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._current_episode = {
            "id": question_id,
            "instruction": instruction,
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "steps": [],
            "done": False,
            "num_steps": 0,
            "error": None,
        }
        if extra:
            self._current_episode["extra"] = extra

        os.makedirs(os.path.join(self.screenshots_dir, question_id), exist_ok=True)

    def _save_screenshot(self, question_id: str, step_idx: int, screenshot: Any) -> Optional[str]:
        if screenshot is None:
            return None

        rel_path = os.path.join("screenshots", question_id, f"step_{step_idx}.png")
        abs_path = os.path.join(self.output_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        if isinstance(screenshot, bytes):
            with open(abs_path, "wb") as f:
                f.write(screenshot)
            raw_bytes = screenshot
        elif "PIL" in str(type(screenshot)):
            screenshot.save(abs_path)
            buf = BytesIO()
            screenshot.save(buf, format="PNG")
            raw_bytes = buf.getvalue()
        else:
            return None

        if self.embed_base64:
            return "data:image/png;base64," + base64.b64encode(raw_bytes).decode("ascii")
        return rel_path.replace("\\", "/")

    def record_step(
        self,
        step_idx: int,
        screenshot: Any,
        model_output: Optional[str],
        action: Optional[str],
        user_question: Optional[str] = None,
        response: Optional[str] = None,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._current_episode is None:
            raise RuntimeError("start_episode() must be called before record_step()")

        question_id = self._current_episode["id"]
        screenshot_ref = self._save_screenshot(question_id, step_idx, screenshot)

        step_data: Dict[str, Any] = {
            "step": step_idx,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "screenshot": screenshot_ref,
            "model_output": model_output,
            "action": action,
        }
        if user_question is not None:
            step_data["user_question"] = user_question
        if response is not None:
            step_data["response"] = response
        if info:
            step_data["info"] = info
        step_data["done"] = done

        self._current_episode["steps"].append(step_data)
        self._current_episode["num_steps"] = len(self._current_episode["steps"])
        if done:
            self._current_episode["done"] = True
        self._flush_current()

    def end_episode(self, error: Optional[str] = None) -> None:
        if self._current_episode is None:
            return
        self._current_episode["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        if error:
            self._current_episode["error"] = error
        self.data["episodes"].append(self._current_episode)
        self._current_episode = None
        self._flush()

    def _flush_current(self) -> None:
        """Persist completed episodes + in-progress episode so crashes don't lose data."""
        payload = dict(self.data)
        episodes = list(self.data["episodes"])
        if self._current_episode is not None:
            episodes = episodes + [self._current_episode]
        payload["episodes"] = episodes
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _flush(self) -> None:
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)


def load_questions(path: str) -> List[Dict[str, Any]]:
    """
    Load questions from JSON.

    Supported formats:
      1) ["question1", "question2"]
      2) {"questions": ["q1", "q2"]}
      3) {"questions": [{"id": "...", "instruction": "...", "config": [...]}]}
      4) [{"id": "...", "instruction": "..."}]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "questions" in raw:
        items = raw["questions"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(
            "Questions file must be a list, or an object with a 'questions' field."
        )

    questions: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            questions.append(
                {
                    "id": f"q{i:04d}",
                    "instruction": item,
                    "config": [],
                }
            )
        elif isinstance(item, dict):
            instruction = item.get("instruction") or item.get("question") or item.get("text")
            if not instruction:
                raise ValueError(f"Question item #{i} missing instruction/question/text")
            qid = item.get("id") or f"q{i:04d}"
            questions.append(
                {
                    "id": str(qid),
                    "instruction": instruction,
                    "config": item.get("config", []),
                    "snapshot": item.get("snapshot"),
                    "related_apps": item.get("related_apps", []),
                }
            )
        else:
            raise ValueError(f"Unsupported question item type at index {i}: {type(item)}")
    return questions


def question_to_task_config(question: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimal DesktopEnv task_config (no real evaluator)."""
    return {
        "id": question["id"],
        "instruction": question["instruction"],
        "config": question.get("config") or [],
        "snapshot": question.get("snapshot") or "collection",
        "related_apps": question.get("related_apps") or [],
        # Dummy evaluator so DesktopEnv._set_task_info works; collection skips evaluate().
        "evaluator": {
            "func": "infeasible",
            "result": {},
        },
    }
