"""Single-episode runner: drives the agent through one task and records evidence.

Per episode the result directory receives step_<i>.png screenshots, a
traj.jsonl step log, result.txt with the rule-based validator score, and a
meta_<task_id>.json holding the full trajectory record.
"""

import copy
import datetime
import json
import logging
import os
import time

logger = logging.getLogger("desktopenv.experiment")


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{get_example_id(example)}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(
        logging.FileHandler(os.path.join(example_result_dir, "runtime.log"))
    )
    return runtime_logger


def example_get(example, key: str, default=None):
    if hasattr(example, "get") and callable(getattr(example, "get")):
        try:
            return example.get(key, default)
        except TypeError:
            return example.get(key)
    return getattr(example, key, default)


def get_example_id(example, fallback: str = "unknown_task") -> str:
    value = example_get(example, "id", fallback)
    return str(value) if value is not None else fallback


def persist_evaluation_result(result, example_result_dir):
    if isinstance(result, dict):
        score = float(result.get("score", 0.0))
        with open(os.path.join(example_result_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False, default=str)
    else:
        score = float(result)

    logger.info("Result: %.2f", score)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{score}\n")
    return score


def absolute_to_relative_coordinate(coords: list, height, width):
    # [abs_x, abs_y] -> [rel_x, rel_y] (0-1000)
    coords[0] = coords[0] / width * 1000
    coords[1] = coords[1] / height * 1000
    return coords


def run_single_example(agent, env, example, max_steps, instruction, args,
                       example_result_dir, scores):
    setup_logger(example, example_result_dir)
    agent.reset()
    env.reset(task_config=example)
    time.sleep(30)  # Wait for the environment to be ready
    obs = env._get_obs()

    done = False
    step_idx = 0
    global_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    example_id = get_example_id(example)
    evaluator = example_get(example, "evaluator", {}) or {}
    related_apps = example_get(example, "related_apps") or [example_get(example, "snapshot")]

    model_name = agent.model if hasattr(agent, "model") else (
        agent.model_name if hasattr(agent, "model_name") else "undefined model")

    meta_json = {
        "trace_id": f"{example_id}_{global_timestamp}",
        "task_id": example_id,
        "platform": "Desktop",
        "subdomain": "Ubuntu 22.04",
        "environment_details": {
            "screen_resolution": f"{args.screen_width}x{args.screen_height}",
            "related_apps": related_apps,
        },
        "instruction": instruction,
        "agent": model_name,
        "annotation_metadata": {
            "annotator_id": "",
            "annotation_tool_version": "",
            "timestamp": global_timestamp,
        },
        # To be populated
        "trajectory": [],
        "trajectory_length": 0,
        "rule_judge": {
            "reward": -1,
        },
        "score": None,
    }

    while not done and step_idx < max_steps:
        current_screenshot_name = f"step_{step_idx}.png"
        current_screenshot_path = os.path.join(example_result_dir, current_screenshot_name)
        with open(current_screenshot_path, "wb") as _f:
            _f.write(obs["screenshot"])

        # The agent loop resizes the observation internally; pass a deep copy
        # so the recorded screenshot stays at native resolution.
        agent_obs = copy.deepcopy(obs)
        response, actions = agent.predict(instruction, agent_obs)

        raw_response = ""
        thought = ""
        meta_action = []
        action_list = []
        coordinate_1 = None
        coordinate_2 = None
        code_result = ""
        last_reward = None
        for _, (action, response_per_action) in enumerate(zip(actions, response)):
            raw_response = response_per_action.get("raw_response", "")
            thought = response_per_action.get("thought", "")
            action_list.append(response_per_action.get("action", ""))
            meta_action.append(response_per_action.get("meta_action", None))
            if response_per_action.get("coordinate", None):
                if not coordinate_1:
                    coordinate_1 = absolute_to_relative_coordinate(
                        response_per_action.get("coordinate"),
                        args.screen_height, args.screen_width)
                elif not coordinate_2:
                    coordinate_2 = [coordinate_1, absolute_to_relative_coordinate(
                        response_per_action.get("coordinate"),
                        args.screen_height, args.screen_width)]
                    coordinate_1 = None

            # Code-tool actions run through the controller before the env step.
            if action.startswith("BASH") or action.startswith("PYTHON"):
                if action.startswith("BASH"):
                    code = action[5:]
                    result = env.controller.run_bash_script(code)
                else:
                    code = action[7:]
                    result = env.controller.run_python_script(code)

                action = code
                code_result += f"Status: {result.get('status', '')}\n"
                code_result += f"Output: {result.get('output', '')}\n"
                code_result += f"Error: {result.get('error', '')}\n"
                agent.last_code_result = code_result

            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            last_reward = reward

            if done:
                logger.info("The episode is done.")
                break

        step_data = {
            "step_index": step_idx,
            "screenshot_path": os.path.relpath(
                current_screenshot_path,
                start=os.path.dirname(example_result_dir),
            ),
            "raw_response": raw_response,
            "thought": thought,
            "action": ";".join(action_list),
            "coordinate": coordinate_1,      # [x, y] normalized 0-1000, or None
            "coordinate2": coordinate_2,     # [[x1,y1], [x2,y2]] or None
            "meta_action": meta_action,
            "code_result": code_result,
        }
        meta_json["trajectory"].append(step_data)

        traj_item = {
            "instruction": instruction,
            "step_num": step_idx + 1,
            "response": response,
            "reward": last_reward,
            "done": done,
            "screenshot_file": current_screenshot_name,
        }
        with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(traj_item, ensure_ascii=False, default=str))
            f.write("\n")

        step_idx += 1

    meta_json["trajectory_length"] = step_idx

    # Rule-based evaluation via the task's validator, when the task carries one.
    need_rule_judge = evaluator.get("need_rule_judge", True) if isinstance(evaluator, dict) else True
    if need_rule_judge:
        rule_evaluate_result = env.evaluate()
        meta_json["rule_judge"]["result"] = rule_evaluate_result
        meta_json["rule_judge"]["reward"] = persist_evaluation_result(
            rule_evaluate_result, example_result_dir)
        meta_json["score"] = meta_json["rule_judge"]["reward"]
        if scores is not None:
            scores.append(meta_json["score"])

    meta_json_path = os.path.join(
        os.path.dirname(example_result_dir), f"meta_{example_id}.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=4, ensure_ascii=False, default=str)
    logger.info(f"Saved meta json to {meta_json_path}")
