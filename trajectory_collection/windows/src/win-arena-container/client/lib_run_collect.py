"""Run agent loop on one question and collect screenshots + model outputs."""
from __future__ import annotations

import datetime
import logging
import time
import traceback
from typing import Any, Dict, Optional

from collection_recorder import CollectionRecorder, question_to_task_config

logger = logging.getLogger("desktopenv.collect")


def run_single_collect(
    agent,
    env,
    question: Dict[str, Any],
    max_steps: int,
    args,
    recorder: CollectionRecorder,
) -> None:
    """
    Execute one question and append an episode to the collection JSON.

    Unlike evaluation, this:
      - records the screenshot the model actually saw (pre-action)
      - stores model_output / action inline in JSON
      - skips env.evaluate()
    """
    instruction = question["instruction"]
    question_id = question["id"]
    example = question_to_task_config(question)

    recorder.start_episode(
        question_id=question_id,
        instruction=instruction,
        extra={"max_steps": max_steps},
    )

    try:
        agent.reset()
        obs = env.reset(task_config=example)
        done = False
        step_idx = 0
        start_time = datetime.datetime.now()

        while not done and step_idx < max_steps:
            if obs is None:
                logger.error("Observation is None. Waiting before next step.")
                time.sleep(5)
                step_idx += 1
                continue

            logger.info("Collect [%s] step %d: Agent thinking...", question_id, step_idx)
            response, actions, logs, computer_update_args = agent.predict(instruction, obs)

            if computer_update_args:
                env.controller.update_computer(**computer_update_args)

            model_output = None
            user_question = None
            if logs:
                model_output = logs.get("plan_result")
                user_question = logs.get("user_question")

            # Screenshot the model used for this prediction (pre-action).
            pre_screenshot = obs.get("screenshot") if obs else None

            if not actions:
                logger.warning("No actions returned; stopping episode.")
                recorder.record_step(
                    step_idx=step_idx,
                    screenshot=pre_screenshot,
                    model_output=model_output,
                    action=None,
                    user_question=user_question if getattr(args, "save_user_question", False) else None,
                    response=response if getattr(args, "save_response", False) else None,
                    done=True,
                    info={"reason": "empty_actions"},
                )
                break

            for action in actions:
                action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                elapsed = datetime.datetime.now() - start_time
                logger.info("Collect [%s] Step %d (%s): %s", question_id, step_idx, action_timestamp, action)

                obs, reward, done, info = env.step(action, args.sleep_after_execution)
                logger.info("Done: %s  Reward: %.2f  Elapsed: %s", done, reward, elapsed)

                recorder.record_step(
                    step_idx=step_idx,
                    screenshot=pre_screenshot,
                    model_output=model_output,
                    action=action,
                    user_question=user_question if getattr(args, "save_user_question", False) else None,
                    response=response if getattr(args, "save_response", False) else None,
                    done=done,
                    info=info or {},
                )

                # Only the first action in a multi-action predict uses this screenshot/output.
                pre_screenshot = obs.get("screenshot") if obs else None
                model_output = None
                user_question = None
                response = None

                if done:
                    logger.info("Episode finished by agent signal.")
                    break

            step_idx += 1

        recorder.end_episode()
        logger.info("Finished collecting question %s (%d steps)", question_id, step_idx)

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error("Exception collecting %s: %s\n%s", question_id, e, error_traceback)
        recorder.end_episode(error=f"{e}\n{error_traceback}")
        raise
