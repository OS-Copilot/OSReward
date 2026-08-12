"""Regression checks for compact collection defaults and prompt isolation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from webtrail.config import Config
from webtrail.prompts import step_block, task_block
from webtrail.recorder import RunRecorder
from webtrail.types import PageState, Task


class CompactDefaultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = Task.from_record({
            "url": "https://example.com/start",
            "instruction": "Find the visible answer.",
            "steps": ["SECRET REFERENCE PATH"],
            "criteria": ["SECRET GOLD ANSWER"],
        }, 0)

    def test_agent_prompt_contains_only_instruction_and_live_url(self) -> None:
        state = PageState(
            url="https://example.com/live",
            title="SECRET PAGE TITLE",
            html="<html></html>",
            screenshot_png=None,
            elements=[],
            axtree={},
            scroll={},
            viewport=(1920, 1080),
        )
        prompt = step_block(self.task, state, 0, 30, [])
        self.assertIn("Find the visible answer.", prompt)
        self.assertIn("https://example.com/live", prompt)
        self.assertNotIn("SECRET REFERENCE PATH", prompt)
        self.assertNotIn("SECRET GOLD ANSWER", prompt)
        self.assertNotIn("SECRET PAGE TITLE", prompt)
        self.assertEqual(
            task_block(self.task),
            "## Task\n\nFind the visible answer.",
        )

    def test_large_artifacts_are_disabled_by_default(self) -> None:
        run = Config().run
        self.assertFalse(run.save_html)
        self.assertFalse(run.save_axtree)
        self.assertFalse(run.save_model_views)
        self.assertFalse(run.annotate_screenshots)

    def test_default_recorder_writes_only_compact_observation(self) -> None:
        image = Image.new("RGB", (8, 8), "white")
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            image.save(handle, format="PNG")
            handle.flush()
            png = Path(handle.name).read_bytes()

        state = PageState(
            url="https://example.com/live",
            title="Example",
            html="<html>large</html>",
            screenshot_png=png,
            elements=[{"role": "button", "name": "Go"}],
            axtree={"role": "WebArea"},
            scroll={},
            viewport=(8, 8),
        )
        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder(directory).open_trajectory(self.task)
            recorder.save_observation(0, state, None, model_view_png=png)
            dirs = {path.name for path in recorder.root.iterdir() if path.is_dir()}
            self.assertEqual(dirs, {"screenshots", "elements", "states"})

    def test_large_artifacts_can_be_enabled_explicitly(self) -> None:
        image = Image.new("RGB", (8, 8), "white")
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            image.save(handle, format="PNG")
            handle.flush()
            png = Path(handle.name).read_bytes()

        state = PageState(
            url="https://example.com/live",
            title="Example",
            html="<html>large</html>",
            screenshot_png=png,
            elements=[],
            axtree={"role": "WebArea"},
            scroll={},
            viewport=(8, 8),
        )
        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder(
                directory,
                save_html=True,
                save_axtree=True,
                save_model_views=True,
            ).open_trajectory(self.task)
            recorder.save_observation(0, state, None, model_view_png=png)
            dirs = {path.name for path in recorder.root.iterdir() if path.is_dir()}
            self.assertEqual(
                dirs,
                {"screenshots", "model_views", "html", "axtree", "elements", "states"},
            )

    def test_example_tasks_have_only_url_and_instruction(self) -> None:
        example = Path(__file__).parents[1] / "tasks" / "example.jsonl"
        for line in example.read_text(encoding="utf-8").splitlines():
            self.assertEqual(set(json.loads(line)), {"url", "instruction"})


if __name__ == "__main__":
    unittest.main()
