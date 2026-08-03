"""Prompt templates for the tool-call collection agent.

Referenced by `model_profiles.py`: one system + user prompt pair per prompt
format ("gemini3", "qwen3vl"). Both define the same `mobile_use` tool schema
and require the output order <thinking> -> <tool_call> -> <conclusion>.
"""

GEMINI3PRO_SYSTEM_PROMPT = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 999x999.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n* `type`: Input the specified text into the activated input box.\n* `answer`: Output the answer.\n* `system_button`: Press the system button.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. The coordinates should be values from 0 to 999. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Additional note:
1. At each step you will be provided with the most recent screenshots (the last 1 or N screenshots).
2. If the user query requires information from the current page, MAKE SURE to explicitly write it in <conclusion> as your memory, because you may not be able to access this interface again in later steps.
3. If you want to dismiss the on-screen keyboard for subsequent actions, use the Back button or click on an empty area; do not use a swipe action.
4. Do not include any specific coordinates in <thinking>.
5. Previous actions may contain mistakes. If you detect an error, correct it or try an alternative solution path.

# Response format

Response format for every step:
1) <thinking>A step-by-step reasoning thought explaining the next move.</thinking>
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.
3) <conclusion>A short summary of the action just taken and all information that will be used in the next step to answer the user query</conclusion>

Rules:
- Output exactly in the order: <thinking>, <tool_call>, <conclusion>.
- In <thinking>, provide a short reasoning paragraph for the next move (can be multiple sentences, but keep it concise).
- In <conclusion>, provide a short summary sentence of the executed action.
- Do not output anything else outside those three parts.
- If finishing, use action=terminate in the tool call.

Example:
<thinking>
I have already found the score for SWE-agent-LM-32B from the previous search results, which is 40.2 on the SWE-Bench Verified benchmark. Now, I need to find the score for SWE-gym-32. The search bar is currently empty and active, so I will type "SWE-gym-32B paper SWE-Bench Verified" and press enter to search for its benchmark results.
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "type", "text": "SWE-gym-32B paper SWE-Bench Verified\n"}}
</tool_call>
<conclusion>
I typed "SWE-gym-32B paper SWE-Bench Verified" into the search bar and pressed enter to search for its benchmark results and successfully found that the score for SWE-agent-LM-32B is 80.2 on the SWE-Bench Verified benchmark.
</conclusion>"""


GEMINI3PRO_USER_PROMPT = """The user query: {instruction}.
Task progress (You have done the following operation on the current device): {history}.

Reminder: Output ONLY in the required format: a <thinking>...</thinking> block followed by a single <tool_call>...</tool_call> block and a <conclusion>...</conclusion> block. Do not output anything else.
"""

# ===========================================================================
# Qwen3VL tool-call prompts
# ===========================================================================

# QWEN3VL_SYSTEM_PROMPT = """

# # Tools

# You may call one or more functions to assist with the user query.

# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 999x999.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n* `type`: Input the specified text into the activated input box.\n* `answer`: Output the answer.\n* `system_button`: Press the system button.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. The coordinates should be values from 0 to 999. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
# </tools>

# For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call>

# # Response format

# Response format for every step:
# 1) Thought: one concise sentence explaining the next move (no multi-step reasoning).
# 2) Action: a short imperative describing what to do in the UI.
# 3) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

# Rules:
# - Output exactly in the order: Thought, Action, <tool_call>.
# - Be brief: one sentence for Thought, one for Action.
# - Do not output anything else outside those three parts.
# - If finishing, use action=terminate in the tool call."""


QWEN3VL_SYSTEM_PROMPT = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 999x999.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n* `type`: Input the specified text into the activated input box.\n* `answer`: Output the answer.\n* `system_button`: Press the system button.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. The coordinates should be values from 0 to 999. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) <thinking>A step-by-step reasoning thought explaining the next move.</thinking>
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.
3) <conclusion>A short summary of the action just taken and all information that will be used in the next step to answer the user query</conclusion>

Rules:
- Output exactly in the order: <thinking>, <tool_call>, <conclusion>.
- In <thinking>, provide a short reasoning paragraph for the next move (can be multiple sentences, but keep it concise).
- In <conclusion>, provide a short summary sentence of the executed action.
- Do not output anything else outside those three parts.
- If finishing, use action=terminate in the tool call.

Example:
<thinking>
I have already found the score for SWE-agent-LM-32B from the previous search results, which is 40.2 on the SWE-Bench Verified benchmark. Now, I need to find the score for SWE-gym-32. The search bar is currently empty and active, so I will type "SWE-gym-32B paper SWE-Bench Verified" and press enter to search for its benchmark results.
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "type", "text": "SWE-gym-32B paper SWE-Bench Verified\n"}}
</tool_call>
<conclusion>
I typed "SWE-gym-32B paper SWE-Bench Verified" into the search bar and pressed enter to search for its benchmark results and successfully found that the score for SWE-agent-LM-32B is 80.2 on the SWE-Bench Verified benchmark.
</conclusion>

"""


# Same as QWEN3VL_SYSTEM_PROMPT, but explicitly tells the agent it will receive multiple most recent screenshots.


QWEN3VL_USER_PROMPT = """The user query: {instruction}.
Task progress (You have done the following operation on the current device): {history}.
"""

# ===========================================================================
# Qwen25VL tool-call prompts
# ===========================================================================
