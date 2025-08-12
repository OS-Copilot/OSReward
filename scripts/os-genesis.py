import os
import re
import json
import uuid
from datetime import datetime

jsonl_path = "os_genesis_web_training.jsonl"
img_dir = "os_genesis_web_screenshots"

last_inst = ""
current_traj = None
results = []

with open(jsonl_path, mode="r", encoding="utf-8") as r:
    json_list = r.read().strip().split("\n")
    for json_entry in json_list:
        step = json.loads(json_entry)
        raw_prompt = step["conversations"][0]["value"].strip()

        inst = [
            item for item in raw_prompt.split("\n\n")
            if item.startswith("Instruction: ")
        ][0][13:]

        raw_a11y_tree = re.search(r'Accessibility tree: ([\S\s]+)Action History', raw_prompt)
        raw_a11y_tree = "" if raw_a11y_tree is None else raw_a11y_tree[1].strip() + "\n"
        a11y_tree_path = step["image"].replace(".png", ".txt")
        with open(os.path.join(img_dir, a11y_tree_path), mode="w", encoding="utf-8") as w:
            w.write(raw_a11y_tree)

        if inst != last_inst:
            last_inst = inst
            if current_traj is not None:
                results.append(current_traj)
            current_traj = {
                "trace_id": str(uuid.uuid4()),
                "task_id": "...",
                "task_source": "OS-Genesis/web-training",
                "in_domain": "0",
                "platform": "Web",
                "subdomain": "...",
                "environment_details": {
                    "screen_resolution": "1280x720",
                    "os_version": "...",
                    "browser_name": "Chrome",
                    "browser_version": "..."
                },
                "instruction": "...",
                "agent_metadata": {
                    "producer": "...",
                    "model_name": "...",
                    "prompt_version": "..."
                },
                "trajectory": [],
                "trajectory_length": 0,
                "orm_label": {
                    "score": "...",
                    "binary_reward": 1,
                    "rationale": "..."
                },
                "annotation_metadata": {
                    "annotation_id": "...",
                    "annotation_tool_version": "...",
                    "timestamp": datetime.now().isoformat()
                }
            }

        raw_response = step["conversations"][1]["value"]
        raw_thought = raw_response.split("In summary")[0]
        raw_action = re.search(r'```(.+)```', raw_response)
        raw_action = "" if raw_action is None else raw_action[1]

        actions = raw_action.split(" ")
        action_name = "" if len(actions) < 1 else actions[0]

        current_traj["trajectory"].append({
            "step_index": len(current_traj["trajectory"]),
            "state": {
                "screenshot_path": step["image"],
                "a11y_tree_path": a11y_tree_path
            },
            "raw_response": raw_response,
            "thought": raw_thought.strip(),
            "action": {
                "type": action_name,
                "coordinates_abs": "",
                "coordinates_rel": "",
                "object_id": actions[1][1:-1] if action_name in ["click", "type", "hover"] else "",
                "content": actions[2][1:-1] if action_name in ["type"] else "",
                "key_comb": actions[1][1:-1] if action_name in ["press"] else "",
                "direction": actions[1][1:-1] if action_name in ["scroll"] else "",
                "tab_index": actions[1][1:-1] if action_name in ["tab_focus"] else "",
                "url": actions[1][1:-1] if action_name in ["goto"] else "",
                "answer": actions[1][1:-1] if action_name in ["stop"] else ""
            },
            "prm_label": {
                "is_error": False,
                "correction": None
            }
        })
        current_traj["trajectory_length"] += 1

results.append(current_traj)
with open("output.jsonl", mode="w", encoding="utf-8") as w:
    w.write("\n".join([json.dumps(item) for item in results]))
