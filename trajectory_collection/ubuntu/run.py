"""Collect agent trajectories on an Ubuntu desktop environment.

Given a task list (OSWorld-style JSON files), a tool-call agent runs each task
step by step inside a VM through any OpenAI-compatible model endpoint; every
step's screenshot, model response, and parsed action are persisted, together
with the task validator's verdict.

Usage:
    python run.py \
        --provider_name docker \
        --rollout_test_all_meta_path tasks/test_small.json \
        --rollout_task_dir tasks/examples \
        --model qwen3-vl-235b-a22b-instruct \
        --result_dir results/demo

The model endpoint comes from --model / --base_url / --api_key or the
MODEL_NAME / MODEL_BASE_URL / MODEL_API_KEY environment variables (a .env file
in this directory is auto-loaded). Re-running with the same --result_dir
resumes and skips finished episodes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import signal
import sys
import time
from multiprocessing import Manager, Process, current_process
from queue import Queue
from typing import Dict, List

import lib_run_single
from agents.toolcall_agent import ToolCallAgent
from desktop_env.desktop_env import DesktopEnv

# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False

# Load environment variables from a .env file next to this script
if os.path.exists(".env"):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect agent trajectories on Ubuntu desktop tasks"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None,
                        help="Path to the VM image; omit to let the provider "
                             "download / locate one.")
    parser.add_argument("--provider_name", type=str, default="docker",
                        choices=["docker", "vmware", "aws"])
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument("--action_space", type=str, default="pyautogui")
    parser.add_argument("--observation_type", choices=["screenshot"], default="screenshot")
    parser.add_argument("--sleep_after_execution", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Cache directory for task setup files.")
    parser.add_argument("--region", type=str, default="us-east-1",
                        help="Region for the AWS provider.")
    parser.add_argument("--client_password", type=str, default="", help="VM user password")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--input_screen_width", type=int, default=1504,
                        help="Width of the screenshot the model sees.")
    parser.add_argument("--input_screen_height", type=int, default=832,
                        help="Height of the screenshot the model sees.")

    # task config
    parser.add_argument("--rollout_test_all_meta_path", type=str,
                        default="tasks/test_all.json",
                        help="JSON mapping domain -> [task_id, ...].")
    parser.add_argument("--rollout_task_dir", type=str, default=None,
                        help="Directory with <domain>/<task_id>.json task files "
                             "(default: alongside the task list).")
    parser.add_argument("--domain", type=str, default="all",
                        help="Collect only this domain of the task list.")

    # model config
    parser.add_argument("--model", type=str,
                        default=os.environ.get("MODEL_NAME", ""))
    parser.add_argument("--base_url", type=str,
                        default=os.environ.get("MODEL_BASE_URL", ""))
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("MODEL_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--use_thinking", action="store_true", default=False)
    parser.add_argument("--max_trajectory_length", type=int, default=8,
                        help="Max text-history steps kept in the model context.")
    parser.add_argument("--max_image_history_length", type=int, default=5,
                        help="Max screenshots kept in the model context.")
    parser.add_argument("--keep_first_image", action="store_true", default=False)
    parser.add_argument("--keep_all_text", action="store_true", default=False)
    parser.add_argument("--enable_code_tool", action="store_true", default=False,
                        help="Expose the bash/python code tool to the agent.")

    # logging / parallelism
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of environments to run in parallel")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    return parser.parse_args()


args = config()
logger = logging.getLogger()
logger.setLevel(getattr(logging, args.log_level.upper()))

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(getattr(logging, args.log_level.upper()))
stdout_handler.setFormatter(logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
))
stdout_handler.addFilter(logging.Filter("desktopenv"))
logger.addHandler(stdout_handler)
logger = logging.getLogger("desktopenv.experiment")


def distribute_tasks(test_all_meta: dict) -> List[tuple]:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks


def task_get(task_config, key: str, default=None):
    if hasattr(task_config, "get") and callable(getattr(task_config, "get")):
        try:
            return task_config.get(key, default)
        except TypeError:
            return task_config.get(key)
    return getattr(task_config, key, default)


def load_offline_task_config(args: argparse.Namespace, domain: str, example_id: str):
    config_file = os.path.join(args.rollout_task_dir, f"{domain}/{example_id}.json")
    with open(config_file, "r", encoding="utf-8") as f:
        example = json.load(f)
    example["id"] = example_id
    return example


def run_env_tasks(task_queue: Queue, args: argparse.Namespace, shared_scores: list):
    global active_environments
    env = None
    try:
        snapshot_name = "init_state"
        if args.provider_name == "aws":
            try:
                from desktop_env.providers.aws.manager import IMAGE_ID_MAP

                screen_size = (args.screen_width, args.screen_height)
                snapshot_name = IMAGE_ID_MAP[args.region].get(
                    screen_size, IMAGE_ID_MAP[args.region][(1920, 1080)]
                )
            except Exception as e:
                logger.error(f"Failed to get snapshot_name from IMAGE_ID_MAP: {e}")
                snapshot_name = "init_state"

        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=args.action_space,
            provider_name=args.provider_name,
            region=args.region,
            snapshot_name=snapshot_name,
            screen_size=(args.screen_width, args.screen_height),
            headless=args.headless,
            os_type="Ubuntu",
            require_a11y_tree=False,
            client_password=args.client_password,
            cache_dir=args.cache_dir,
        )
        env.start()
        active_environments.append(env)

        agent = ToolCallAgent(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            max_trajectory_length=args.max_trajectory_length,
            keep_first_image=args.keep_first_image,
            keep_all_text=args.keep_all_text,
            use_thinking=args.use_thinking,
            input_screen_size=(args.input_screen_width, args.input_screen_height),
            enable_code_tool=args.enable_code_tool,
        )

        logger.info(f"Process {current_process().name} started.")
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            domain, example_id = item
            try:
                example = load_offline_task_config(args, domain, example_id)
                instruction = task_get(example, "instruction")
                logger.info(f"[{current_process().name}][Domain]: {domain}")
                logger.info(f"[{current_process().name}][Example ID]: {example_id}")
                logger.info(f"[{current_process().name}][Instruction]: {instruction}")
                example_result_dir = os.path.join(args.result_dir, domain, example_id)
                os.makedirs(example_result_dir, exist_ok=True)
                try:
                    lib_run_single.run_single_example(
                        agent,
                        env,
                        example,
                        args.max_steps,
                        instruction,
                        args,
                        example_result_dir,
                        shared_scores,
                    )
                except Exception as e:
                    logger.error(f"Exception in {domain}/{example_id}: {e}", exc_info=True)
                    error_path = os.path.join(example_result_dir, "error.txt")
                    with open(error_path, "a", encoding="utf-8") as f:
                        f.write(f"{datetime.datetime.now().isoformat()}: {e}\n")
            except Exception as e:
                logger.error(f"Failed to load task {domain}/{example_id}: {e}", exc_info=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")


def signal_handler(signum, frame):
    """Handle termination signals (SIGINT, SIGTERM) to gracefully shutdown environments."""
    global is_terminating, active_environments, processes

    if is_terminating:
        return
    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")

    for env in active_environments:
        try:
            env.close()
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    for p in processes:
        if p is not None and p.is_alive():
            try:
                p.terminate()
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
    sys.exit(0)


def get_unfinished_tasks(test_file_list: Dict, result_dir) -> Dict:
    """An episode counts as finished when its meta_<task_id>.json exists."""
    unfinished = {}
    for domain, task_list in test_file_list.items():
        logger.info(f"[Origin {domain} task nums]: {len(task_list)}")
        if os.path.exists(os.path.join(result_dir, domain)):
            for task_id in task_list:
                task_dir = os.path.join(result_dir, domain, task_id)
                meta_json_path = os.path.join(result_dir, domain, f"meta_{task_id}.json")
                if not os.path.exists(meta_json_path):
                    unfinished.setdefault(domain, []).append(task_id)
                    shutil.rmtree(path=task_dir, ignore_errors=True)
            logger.info(f"[Unfinished {domain} task nums]: {len(unfinished.get(domain, []))}")
        else:
            unfinished[domain] = task_list
    return unfinished


def offline_test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    all_tasks = distribute_tasks(test_all_meta)
    logger.info(f"Total tasks: {len(all_tasks)}")
    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)
        processes = []
        for i in range(args.num_envs):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores),
                name=f"EnvProcess-{i + 1}",
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started process {p.name} with PID {p.pid}")
        try:
            while True:
                alive_count = 0
                for idx, p in enumerate(processes):
                    if not p.is_alive():
                        logger.warning(f"Process {p.name} died, restarting...")
                        new_p = Process(
                            target=run_env_tasks,
                            args=(task_queue, args, shared_scores),
                            name=f"EnvProcess-Restart-{idx + 1}",
                        )
                        new_p.daemon = True
                        new_p.start()
                        processes[idx] = new_p
                    else:
                        alive_count += 1
                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                time.sleep(5)
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt. Shutting down...")
            raise
        scores = list(shared_scores)
    logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if not args.model:
            raise SystemExit("No model configured. Pass --model or set MODEL_NAME.")
        args.rollout_task_dir = args.rollout_task_dir or os.path.dirname(
            args.rollout_test_all_meta_path)

        if args.exp_name:
            args.result_dir = os.path.join(args.result_dir, args.exp_name)
        path_to_args = os.path.join(args.result_dir, "args.json")
        os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
        saved_args = {k: v for k, v in vars(args).items() if k != "api_key"}
        with open(path_to_args, "w", encoding="utf-8") as f:
            json.dump(saved_args, f, indent=4)

        with open(args.rollout_test_all_meta_path, "r", encoding="utf-8") as f:
            test_file_list = json.load(f)
        if args.domain != "all":
            test_file_list = {args.domain: test_file_list[args.domain]}
        test_file_list = get_unfinished_tasks(test_file_list, args.result_dir)
        offline_test(args, test_file_list)
    except KeyboardInterrupt:
        logger.info("Main process received KeyboardInterrupt.")
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}", exc_info=True)
        signal_handler(signal.SIGTERM, None)
    finally:
        for env in active_environments:
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error during final environment cleanup: {e}")
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    p.terminate()
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")
