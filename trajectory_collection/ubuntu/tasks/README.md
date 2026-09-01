# Task files

Tasks follow the OSWorld JSON format. Two inputs drive a collection run:

1. **Task list** (`--rollout_test_all_meta_path`): a JSON object mapping
   domain to task ids.

   ```json
   {
     "libreoffice_calc": ["a01fe23b-1234-5678-9abc-def012345678"],
     "vlc": ["b12fe34c-2345-6789-abcd-ef0123456789"]
   }
   ```

2. **Task directory** (`--rollout_task_dir`, default: alongside the task
   list): one `<domain>/<task_id>.json` per task.

   ```json
   {
     "id": "a01fe23b-1234-5678-9abc-def012345678",
     "snapshot": "libreoffice_calc",
     "instruction": "Rename Sheet1 to Expenses and save the file.",
     "source": "authored",
     "config": [
       {
         "type": "download",
         "parameters": {
           "files": [
             {
               "url": "https://example.com/expenses.xlsx",
               "path": "/home/user/Desktop/expenses.xlsx"
             }
           ]
         }
       },
       { "type": "open", "parameters": { "path": "/home/user/Desktop/expenses.xlsx" } }
     ],
     "related_apps": ["libreoffice_calc"],
     "evaluator": {
       "func": "<a metric from desktop_env/evaluators/metrics/>",
       "expected": { "type": "rule", "rules": { "...": "..." } },
       "result": { "type": "vm_file", "path": "/home/user/Desktop/expenses.xlsx", "dest": "expenses.xlsx" }
     }
   }
   ```

`config` entries prepare the VM state before the agent starts (see
`desktop_env/controllers/setup.py` for the supported types); `evaluator`
defines the rule-based validator (see `desktop_env/evaluators/`). A task
without a reliable validator can set `"evaluator": {"func": "infeasible",
"need_rule_judge": false}`; its reward then comes from the annotate stage or a
VLM judge downstream.

For ready-made task sets in this format, see the
[OSWorld](https://github.com/xlang-ai/OSWorld) evaluation examples.
