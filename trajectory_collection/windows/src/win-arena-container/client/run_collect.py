"""Collect agent trajectories (screenshots + model outputs) into one JSON file.

Usage (inside the client container / client dir):
  python run_collect.py --questions_path collection_examples/questions.json
  python run_collect.py --question "Open Notepad and type hello world"
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import signal
import sys
import traceback
from threading import Event

import requests
from tqdm import tqdm

import lib_run_collect
from collection_recorder import CollectionRecorder, load_questions
from desktop_env.envs.desktop_env import DesktopEnv
from mm_agents.navi.agent import NaviAgent

print("Waiting for the server to start...")

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.propagate = True
datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)


def setup_logging(args):
    logging_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(logging_dir, f"collect-{datetime_str}.log"), encoding="utf-8"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(logging.Filter("desktopenv"))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)


logger = logging.getLogger("desktopenv.collect")


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect WAA agent trajectories (screenshot + model output) into JSON"
    )

    # input
    parser.add_argument(
        "--questions_path",
        type=str,
        default=None,
        help="Path to a JSON file listing questions to collect",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question string (alternative to --questions_path)",
    )
    parser.add_argument(
        "--question_id",
        type=str,
        default=None,
        help="Optional id when using --question",
    )

    # output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./collection_results",
        help="Directory for screenshots + collection JSON",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path of aggregated JSON (default: <output_dir>/collection.json)",
    )
    parser.add_argument(
        "--embed_base64",
        action="store_true",
        help="Embed screenshots as base64 in JSON instead of file paths",
    )
    parser.add_argument(
        "--save_user_question",
        action="store_true",
        help="Also save the prompt sent to the model in each step",
    )
    parser.add_argument(
        "--save_response",
        action="store_true",
        help="Also save raw agent response field in each step",
    )

    # environment / agent (aligned with run.py)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--action_space", type=str, default="pyautogui")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="a11y_tree",
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1200)
    parser.add_argument("--sleep_after_execution", type=float, default=3)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--a11y_backend", type=str, default="uia")
    parser.add_argument("--agent_name", type=str, default="navi")
    parser.add_argument("--som_origin", type=str, default="oss")
    parser.add_argument("--emulator_ip", type=str, default="20.20.20.21")
    parser.add_argument("--model", type=str, default="gpt-4-vision-preview")
    parser.add_argument("--temperature", type=float, default=1.0)

    args, _ = parser.parse_known_args()
    if not args.questions_path and not args.question:
        parser.error("Provide --questions_path or --question")
    if args.output_json is None:
        args.output_json = os.path.join(args.output_dir, "collection.json")
    return args


def build_agent(args):
    if args.agent_name == "navi":
        if args.som_origin in ["a11y", "omni", "mixed-omni"]:
            som_config = None
        elif args.som_origin in ["oss", "mixed-oss"]:
            som_config = {
                "pipeline": ["webparse", "groundingdino", "ocr"],
                "groundingdino": {"prompts": ["icon", "image"]},
                "ocr": {"class_name": "TesseractOCR"},
                "webparse": {"cdp_url": f"http://{args.emulator_ip}:9222"},
            }
        else:
            som_config = None
        return NaviAgent(
            server="oai",
            model=args.model,
            som_config=som_config,
            som_origin=args.som_origin,
            temperature=args.temperature,
        )
    if args.agent_name == "claude":
        from mm_agents.claude.agent import ClaudeAgent

        return ClaudeAgent()
    raise ValueError(f"Unknown agent name: {args.agent_name}")


def resolve_questions(args):
    if args.question:
        qid = args.question_id or "q0000"
        return [{"id": qid, "instruction": args.question, "config": []}]
    return load_questions(args.questions_path)


exit_event = Event()


def quit(signo, _frame):
    print("Interrupted by %d, shutting down" % signo)
    exit_event.set()
    exit(0)


def wait_for_server(ip, port=5000):
    while not exit_event.is_set():
        try:
            response = requests.get(f"http://{ip}:{port}/probe", timeout=7)
            print("Response from server:", response.json())
            break
        except Exception as e:
            print("Failed to get hello:", e)
            print("Retrying...")
            exit_event.wait(5)


for sig in ("TERM", "HUP", "INT"):
    signal.signal(getattr(signal, "SIG" + sig), quit)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args)

    wait_for_server(args.emulator_ip)

    questions = resolve_questions(args)
    logger.info("Will collect %d question(s) -> %s", len(questions), args.output_json)

    recorder = CollectionRecorder(
        output_dir=args.output_dir,
        output_json=args.output_json,
        model=args.model,
        embed_base64=args.embed_base64,
        metadata={
            "observation_type": args.observation_type,
            "action_space": args.action_space,
            "agent_name": args.agent_name,
            "som_origin": args.som_origin,
            "max_steps": args.max_steps,
            "questions_path": args.questions_path,
        },
    )

    agent = build_agent(args)
    env = DesktopEnv(
        action_space=agent.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        require_a11y_tree=args.observation_type
        in ["a11y_tree", "screenshot_a11y_tree", "som"],
        emulator_ip=args.emulator_ip,
        a11y_backend=args.a11y_backend,
    )

    for question in tqdm(questions, desc="Collect"):
        logger.info("[Question ID]: %s", question["id"])
        logger.info("[Instruction]: %s", question["instruction"])
        try:
            lib_run_collect.run_single_collect(
                agent, env, question, args.max_steps, args, recorder
            )
        except Exception as e:
            logger.error("Failed question %s: %s", question["id"], e)
            logger.error(traceback.format_exc())
            # episode already closed with error inside run_single_collect on raise;
            # if raise happened before end_episode, ensure closure
            if recorder._current_episode is not None:
                recorder.end_episode(error=str(e))

    env.close()
    logger.info("Collection complete. JSON: %s", args.output_json)
    logger.info("Episodes: %d", len(recorder.data["episodes"]))


if __name__ == "__main__":
    main()
