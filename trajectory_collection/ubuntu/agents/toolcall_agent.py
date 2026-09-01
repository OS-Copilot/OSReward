import base64
import copy
import json
import logging
import re
import subprocess
import textwrap
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx
import backoff
import openai
from openai import OpenAI
from PIL import Image
from requests.exceptions import SSLError

from agents.qwen_vl_utils import (
    QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_FOR_OSWORLD_INFERENCE,
    QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_FOR_OSWORLD_INFERENCE_WITHOUT_CODE,
    smart_resize,
    QWEN3VL_COMPUTER_USE_TOOL_SCHEMA,
    QWEN3VL_COMPUTER_USE_TOOL_SCHEMA_WITHOUT_CODE,
)
from agents.base import ComputerUseBaseAgent


logger = logging.getLogger("desktopenv.agent")

MAX_RETRY_TIMES = 5
EMPTY_TOOL_CALL_RETRY_TIMES = 5


def encode_image(image_content, input_width: Optional[int] = None, input_height: Optional[int] = None):
    if input_width is not None and input_height is not None:
        image = Image.open(BytesIO(image_content))
        image = image.resize((input_width, input_height))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_content = buffer.getvalue()
    return base64.b64encode(image_content).decode("utf-8")


# def process_image(image_bytes):
#     """Process an image for Qwen VL models."""
#     image = Image.open(BytesIO(image_bytes))
#     width, height = image.size

#     resized_height, resized_width = smart_resize(
#         height=height,
#         width=width,
#         factor=32,
#         max_pixels=16 * 16 * 4 * 1280,
#     )

#     image = image.resize((resized_width, resized_height))

#     buffer = BytesIO()
#     image.save(buffer, format="PNG")
#     processed_bytes = buffer.getvalue()

#     return base64.b64encode(processed_bytes).decode("utf-8")


class ToolCallAgent(ComputerUseBaseAgent):

    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "qwen3-vl",
        base_url: str = "",
        api_key: str = "",
        max_tokens: int = 150000,
        top_p: float = 0.9,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        max_trajectory_length: int = 8,
        add_thought_prefix: bool = False,
        coordinate_type: str = "relative",
        keep_first_image: bool = True,
        keep_all_text: bool = True, # 是否保留全部步数的模型输出（False 退化为 last k）
        keep_cot: bool = True, # 模型输出是否仅保留action/cot+action
        use_thinking: bool = False,
        enable_code_tool: bool = False,
        input_screen_size: tuple = (1920, 1080),
    ):
        self.platform = platform
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.keep_first_image = keep_first_image
        self.keep_cot = keep_cot
        self.add_thought_prefix = add_thought_prefix
        self.coordinate_type = coordinate_type
        self.use_thinking = use_thinking
        self.keep_all_text = keep_all_text
        self.input_screen_size = input_screen_size
        assert action_space in ["pyautogui"], "Invalid action space"
        assert observation_type in ["screenshot"], "Invalid observation type"


        # 为了执行 code 设置的变量
        self.last_code_result: Optional[str] = None
        self.code_results_history: List[str] = []

        # 统一维护对话历史（system + user + assistant + tool）
        # 直接沿用 OpenAI/vLLM 的 messages 协议结构
        self.system_prompt = QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_FOR_OSWORLD_INFERENCE if enable_code_tool else QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_FOR_OSWORLD_INFERENCE_WITHOUT_CODE
        self.messages: List[Dict[str, Any]] = []

        # 记录上一轮产生的 tool_calls，供下一轮填充 tool 结果
        self.pending_tool_calls: List[Any] = []

        self.enable_code_tool = enable_code_tool

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=6000.0
        )

    @staticmethod
    def _py_string(text: str) -> str:
        return json.dumps("" if text is None else str(text), ensure_ascii=False)

    @staticmethod
    def _inject_terminal_tool_call_from_content(response_message: Dict[str, Any]) -> Dict[str, Any]:
        if response_message.get("tool_calls"):
            return response_message

        content = str(response_message.get("content") or "")
        content_lower = content.lower()
        if re.search(r"\bfail(?:ure)?\b", content_lower):
            status = "failure"
        elif re.search(r"\bdone\b", content_lower):
            status = "success"
        else:
            return response_message

        response_message = copy.deepcopy(response_message)
        response_message["tool_calls"] = [
            {
                "id": "call_auto_terminal",
                "function": {
                    "arguments": json.dumps(
                        {"action": "terminate", "status": status},
                        ensure_ascii=False,
                    ),
                    "name": "custom_computer_use",
                },
                "type": "function",
                "index": 0,
            }
        ]
        return response_message

    @staticmethod
    def _sanitize_messages_for_llm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized_messages: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            sanitized: Dict[str, Any] = {"role": role}

            if role == "assistant":
                if "content" in message:
                    sanitized["content"] = message.get("content")
                tool_calls = []
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function") or {}
                    arguments = function.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    tool_calls.append(
                        {
                            "id": tool_call.get("id", "tool_call_0"),
                            "type": "function",
                            "function": {
                                "name": function.get("name", "custom_computer_use"),
                                "arguments": arguments,
                            },
                        }
                    )
                if tool_calls:
                    sanitized["tool_calls"] = tool_calls
            elif role == "tool":
                sanitized["tool_call_id"] = message.get("tool_call_id", "tool_call_0")
                sanitized["content"] = message.get("content", "")
            else:
                sanitized["content"] = message.get("content", "")

            sanitized_messages.append(sanitized)
        return sanitized_messages
    
    def predict(self, instruction: str, obs: Dict) -> Tuple[List[Dict], List[str]]:
        """Predict the next action(s) based on the current observation.

        Returns:
            response_list (List[Dict]): Structured metadata for logging.
            action_list (List[str]): Executable PyAutoGUI code.
        """
        screenshot_bytes = obs["screenshot"]

        # width 一定等于 1920, height 一定等于 1080
        image = Image.open(BytesIO(screenshot_bytes))
        width, height = image.size

        # Resize 到指定分辨率
        processed_image = encode_image(screenshot_bytes, input_width=self.input_screen_size[0], input_height=self.input_screen_size[1])

        # feed tool result for previous tool_calls
        result_text = ""
        if self.pending_tool_calls:
            for tool_call in self.pending_tool_calls:
                try:
                    arguments = json.loads(tool_call["function"]["arguments"]) # vllm
                    name = arguments["action"]
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败: {e}")
                    name = ""
                if name == "code" and self.last_code_result is not None:
                    result_text = f"Code Execution Result:\n```\n{self.last_code_result}\n```"
                    self.last_code_result = None
                elif name == "":
                    result_text = "Fail to parse tool! The output on previous step is NOT a valid JSON object"
                else:
                    result_text = "Success"
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", "tool_call_0"),
                        "content": result_text,
                    }
                )
            # 已经把上一轮的 tool 结果补完，本轮重新收集
            self.pending_tool_calls = []

        # 当前轮 user 消息
        curr_user_content = []

        # 第一次调用 predict 时，初始化 system 附带原始 instruction
        if not self.messages:
            self.messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                    ],
                }
            ]
            instruction_prompt = (
                "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
                f"Instruction: {instruction}"
            )
            curr_user_content.append({"type": "text", "text": instruction_prompt})

        curr_user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{processed_image}"},
            }
        )
        self.messages.append(
            {
                "role": "user",
                "content": curr_user_content,
            }
        )
        self._cleanup_old_context()

        # 让 call_llm 返回原始 message 对象（包含 message.tool_calls 和结构化 content）
        # 如果解析不出来工具就重试
        response_message = self._call_llm_with_tool_call_retry(
            {
                "model": self.model,
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature,
            },
            self.model,
        )

        logger.info(f"Qwen3VL Output: {response_message}")
        if self.keep_cot:
            self.messages.append(response_message)
        else:
            tmp_response = copy.deepcopy(response_message)
            tmp_response.pop("content", None)
            tmp_response.pop("reasoning_content", None)
            self.messages.append(tmp_response)

        # 记录本轮产生的 tool_calls，供下一轮填充 tool 结果
        tool_calls = response_message.get("tool_calls") or []
        self.pending_tool_calls = list(tool_calls)

        # 把工具调用转成我们现有的 meta_data / pyautogui_code
        # 这里不再依赖字符串匹配，response_str 只作为 thought 文本
        content = response_message.get("content", "") # 解析除 <tool_call></tool_call> block
        reasoning_content = response_message.get("reasoning_content", "") # 解析 <think></think> block
        response_str = (("<think>" + reasoning_content + "</think>") if reasoning_content else "") + content

        meta_data, pyautogui_code = self.parse_response(
            response_message,
            response_str,
            width,
            height
        )

        if not pyautogui_code:
            fail_action = {
                "name": "done",
                "command": "DONE",
                "action_type": "DONE",
            }
            meta_data = [{
                "raw_response": f"No tool call\n {response_message}",
                "thought": "Completed",
                "action": "DONE",
                "meta_action": fail_action,
                "coordinate": None,
            }]
            pyautogui_code = ["DONE"]

        logger.info(f"Pyautogui code: {pyautogui_code}")
        # self.debug_print_messages()

        return meta_data, pyautogui_code

    def _cleanup_old_context(self):
        """在 self.messages 上清理超出配额的截图和历史文本。"""
        user_indices_with_img = []
        for idx, msg in enumerate(self.messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            has_image = any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in content
            )
            if has_image:
                user_indices_with_img.append(idx)

        if not user_indices_with_img:
            return

        keep_indices = set()

        # 保留第一条带图 user（如需要）
        if self.keep_first_image:
            first_user_idx = None
            for idx, msg in enumerate(self.messages):
                if msg.get("role") == "user":
                    first_user_idx = idx
                    break
            if first_user_idx is not None:
                content = self.messages[first_user_idx].get("content")
                if isinstance(content, list):
                    has_image = any(
                        isinstance(part, dict) and part.get("type") == "image_url"
                        for part in content
                    )
                    if has_image:
                        keep_indices.add(first_user_idx)

        # 从后往前保留最近 history_n 条带图 user
        remaining = self.max_trajectory_length
        for idx in reversed(user_indices_with_img):
            if idx in keep_indices:
                continue
            if remaining <= 0:
                break
            keep_indices.add(idx)
            remaining -= 1

        # 清理不在 keep_indices 里的截图/文本
        for idx in user_indices_with_img:
            if idx in keep_indices:
                continue
            msg = self.messages[idx]
            content = msg.get("content", [])
            new_content = []
            screenshot_removed = False
            text_removed = False
            for part in content:
                if not isinstance(part, dict):
                    new_content.append(part)
                    continue
                if part.get("type") == "image_url":
                    screenshot_removed = True
                    continue
                if part.get("type") == "text" and not self.keep_all_text:
                    text_removed = True
                    continue
                new_content.append(part)

            placeholders = []
            if screenshot_removed:
                placeholders.append({
                    "type": "text",
                    "text": "[Old Screenshot Removed]",
                })
            if text_removed:
                placeholders.append({
                    "type": "text",
                    "text": "[Old Text Removed]",
                })

            msg["content"] = placeholders + new_content if (placeholders or new_content) else [
                {
                    "type": "text",
                    "text": "[Old Context Removed]",
                }
            ]

    def parse_response(
        self,
        response_message: Dict[str, Any],
        thought: str,
        original_width: int = None,
        original_height: int = None,
    ) -> Tuple[List[Dict], List[str]]:
        """Parse LLM response (dict with tool_calls) and convert it to metadata and pyautogui code.

        - thought: 来自 message.content 提取的自然语言思考文本
        - response_message: 完整的 message dict，包含 tool_calls
        """

        pyautogui_code: List[str] = []
        meta_data: List[Dict] = []

        tool_calls = response_message.get("tool_calls") or []
        if not tool_calls:
            return meta_data, pyautogui_code

        def adjust_coordinates(x: float, y: float) -> Tuple[int, int]:
            x_scale = original_width / 999
            y_scale = original_height / 999
            return int(x * x_scale), int(y * y_scale)

        def make_raw_response() -> str:
            """把 thought + tool_calls 转成原来的 raw_response 兼容格式.

            形如：
            <think>...</think>
            <tool>{...}</tool>
            <tool>{...}</tool>
            """

            parts: List[str] = []
            if thought:
                parts.append(f"{thought}")

            import json as _json

            for tc in tool_calls:
                # tc 可能是 pydantic 转成的 dict
                parts.append(f"<tool_call>{_json.dumps(tc, ensure_ascii=False)}</tool_call>")

            return "\n".join(parts)

        raw_response = make_raw_response()

        def make_meta(action_dict: Dict[str, Any], coordinate: Optional[List[Any]] = None) -> Dict:
            return {
                "raw_response": raw_response,
                "thought": thought,
                "action": action_dict.get("command", ""),
                "meta_action": action_dict,
                "coordinate": coordinate,
            }

        for tool_call in tool_calls:
            try:
                args = tool_call.get("function", {}).get("arguments", "")
                args = json.loads(args)
            except Exception as e:
                logger.error("Invalid tool_call structure: %s", e)
                continue
            
            action = args.get("action")
            if not action:
                continue

            # 每个 tool_call 只生成一条长的 pyautogui 代码，保证与 meta_data 一一对应
            step_code: str = ""
            coord: Optional[List[Any]] = None

            if action in ["left_click", "click"]:
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = adjust_coordinates(x, y)
                    coord = [adj_x, adj_y]
                    step_code = f"pyautogui.click({adj_x}, {adj_y})"
                else:
                    step_code = "pyautogui.click(0, 0)"

            elif action == "right_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = adjust_coordinates(x, y)
                    coord = [adj_x, adj_y]
                    step_code = f"pyautogui.rightClick({adj_x}, {adj_y})"
                else:
                    step_code = "pyautogui.rightClick(0, 0)"

            elif action == "middle_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = adjust_coordinates(x, y)
                    coord = [adj_x, adj_y]
                    step_code = f"pyautogui.middleClick({adj_x}, {adj_y})"
                else:
                    step_code = "pyautogui.middleClick(0, 0)"

            elif action == "double_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = adjust_coordinates(x, y)
                    coord = [adj_x, adj_y]
                    step_code = f"pyautogui.doubleClick({adj_x}, {adj_y})"
                else:
                    step_code = "pyautogui.doubleClick(0, 0)"
            
            elif action == "triple_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = adjust_coordinates(x, y)
                    coord = [adj_x, adj_y]
                    step_code = f"pyautogui.click({adj_x}, {adj_y}, clicks=3)"
                else:
                    step_code = "pyautogui.click(0, 0, clicks=3)"

            elif action == "type":
                text = args.get("text", "")
                step_code = f"pyautogui.write({repr(text)})"

            elif action == "key":
                keys = args.get("keys", [])
                if isinstance(keys, list):
                    cleaned_keys = []
                    for key in keys:
                        if isinstance(key, str):
                            key = key.strip()
                        cleaned_keys.append(key)
                    keys = cleaned_keys

                keys_str = ", ".join([f"'{key}'" for key in keys])
                if len(keys) > 1:
                    step_code = f"pyautogui.hotkey({keys_str})"
                elif len(keys) == 1:
                    step_code = f"pyautogui.press({keys_str})"

            elif action == "scroll":
                # Modify: Fix Coordinate
                pixels = args.get("pixels", 5) # 默认 5
                step_code = ""
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = adjust_coordinates(x, y)
                    coord = [adj_x, adj_y]
                    step_code += f"pyautogui.moveTo({adj_x}, {adj_y});"
                step_code += f"pyautogui.scroll({pixels})"

            elif action == "hscroll":
                pixels = args.get("pixels", 5) # 默认 5
                step_code = ""
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = adjust_coordinates(x, y)
                    coord = [adj_x, adj_y]
                    step_code += f"pyautogui.moveTo({adj_x}, {adj_y});"
                step_code += f"pyautogui.hscroll({pixels})"

            elif action == "wait":
                time = args.get("time", 5)
                step_code = f"time.sleep({time})"

            elif action == "terminate":
                status = args.get("status", "success") # success / failure
                step_code = "DONE" if status == "success" else "FAIL"

            elif action == "answer":
                step_code = "DONE"

            elif action == "mouse_move":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = adjust_coordinates(x, y)
                    coord = [adj_x, adj_y]
                    step_code = f"pyautogui.moveTo({adj_x}, {adj_y})"
                else:
                    step_code = "pyautogui.moveTo(0, 0)"

            elif action == "left_click_drag":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = adjust_coordinates(x, y)
                    coord = [adj_x, adj_y]
                    step_code = f"pyautogui.dragTo({adj_x}, {adj_y}, duration=0.5)"
                else:
                    step_code = "pyautogui.dragTo(0, 0)"

            elif action == "code":
                code_content = args.get("execute_code", "")
                language = args.get("language", "python")
                step_code = f"{language.upper()}|{code_content}"

            # 汇总本次 tool_call 的全部指令为一条长代码，使用 '; ' 连接
            if not step_code:
                continue

            meta_action = {
                "name": action,
                "input": args,
                "id": tool_call.get("id", ""),
                "action_type": tool_call.get("type", "tool_call"),
                "command": step_code,
                "coordinate": coord
            }

            pyautogui_code.append(step_code)
            meta_data.append(make_meta(meta_action, coord))

        return meta_data, pyautogui_code

    def _call_llm_with_tool_call_retry(self, payload: Dict[str, Any], model: str) -> Dict[str, Any]:
        response_message = {}
        supported_actions = {
            "left_click",
            "click",
            "right_click",
            "middle_click",
            "double_click",
            "triple_click",
            "type",
            "key",
            "scroll",
            "hscroll",
            "wait",
            "terminate",
            "mouse_move",
            "left_click_drag",
            "code",
            "answer" # = DONE
        }

        def parse_tool_calls_from_content(content: str) -> tuple[list, str]:
            """
            Fallback: 从 content 文本里用正则提取 <tool_call>...</tool_call>
            返回标准 OpenAI/vLLM 风格的 tool_calls 列表和清理后的 content。
            """
            tool_calls = []
            pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            matches = re.findall(pattern, content, re.DOTALL)

            def validate_code_action(language: str, code: str) -> bool:
                try:
                    if language == "python":
                        compile(code, "<fallback_execute_code>", "exec")
                        return True
                    if language == "bash":
                        proc = subprocess.run(
                            ["bash", "-n"],
                            input=code,
                            text=True,
                            capture_output=True,
                        )
                        return proc.returncode == 0
                except Exception:
                    return False
                return False

            def recover_code_tool_call(raw: str, idx: int) -> Optional[Dict[str, Any]]:
                if '"action"' not in raw or '"code"' not in raw or '"execute_code"' not in raw:
                    return None

                function_name = "custom_computer_use"
                function_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
                if function_match:
                    function_name = function_match.group(1)

                action_match = re.search(r'"action"\s*:\s*"code"', raw)
                language_match = re.search(r'"language"\s*:\s*"(python|bash)"', raw)
                execute_match = re.search(r'"execute_code"\s*:\s*"', raw)
                if not action_match or not language_match or not execute_match:
                    return None

                language = language_match.group(1)
                start = execute_match.end()
                tail = raw[start:]
                terminator_match = re.search(r'"\s*}[\s}]*$', tail, re.DOTALL)
                if not terminator_match:
                    terminator_match = re.search(r'"\s*,\s*"[^"]+"\s*:', tail, re.DOTALL)
                if not terminator_match:
                    return None

                code = tail[:terminator_match.start()]
                if not validate_code_action(language, code):
                    logger.info(
                        "Can't recovered code tool_call via regex fallback on block %s with language=%s",
                        idx,
                        language,
                    )
                    return None

                logger.info(
                    "Recovered code tool_call via regex fallback on block %s with language=%s",
                    idx,
                    language,
                )
                return {
                    "id": f"fallback-tool-call-{idx}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(
                            {
                                "action": "code",
                                "language": language,
                                "execute_code": code,
                            },
                            ensure_ascii=False,
                        ),
                    },
                }

            for idx, match in enumerate(matches):
                try:
                    parsed = json.loads(match)
                except json.JSONDecodeError as e:
                    recovered_tool_call = recover_code_tool_call(match, idx)
                    if recovered_tool_call is not None:
                        tool_calls.append(recovered_tool_call)
                        continue
                    logger.warning(f"Failed to decode fallback tool_call block: {e}; raw={match}")
                    continue

                if isinstance(parsed, dict) and isinstance(parsed.get("function"), dict):
                    function_block = parsed["function"]
                    function_name = function_block.get("name")
                    function_arguments = function_block.get("arguments")

                    if isinstance(function_arguments, dict):
                        function_arguments = json.dumps(function_arguments, ensure_ascii=False)

                    if function_name and isinstance(function_arguments, str):
                        tool_calls.append(
                            {
                                "id": parsed.get("id", f"fallback-tool-call-{idx}"),
                                "type": parsed.get("type", "function"),
                                "function": {
                                    "name": function_name,
                                    "arguments": function_arguments,
                                },
                            }
                        )
                    else:
                        logger.warning(f"Invalid fallback tool_call function block: {parsed}")
                    continue

                function_name = parsed.get("name") if isinstance(parsed, dict) else None
                function_arguments = parsed.get("arguments") if isinstance(parsed, dict) else None
                if isinstance(function_arguments, dict):
                    function_arguments = json.dumps(function_arguments, ensure_ascii=False)

                if function_name and isinstance(function_arguments, str):
                    tool_calls.append(
                        {
                            "id": f"fallback-tool-call-{idx}",
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": function_arguments,
                            },
                        }
                    )
                else:
                    logger.warning(f"Unsupported fallback tool_call format: {parsed}")

            clean_content = re.sub(pattern, '', content, flags=re.DOTALL)
            clean_content = re.sub(r'\n{3,}', '\n\n', clean_content).strip()

            return tool_calls, clean_content

        def extract_supported_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            valid_tool_calls: List[Dict[str, Any]] = []
            for tool_call in tool_calls:
                try:
                    args = json.loads(tool_call.get("function", {}).get("arguments", ""))
                except Exception as e:
                    logger.warning(
                        f"Failed to parse tool_call arguments during retry precheck: {e}; tool_call={tool_call}"
                    )
                    continue

                action = args.get("action")
                if action in supported_actions:
                    valid_tool_calls.append(tool_call)
                else:
                    logger.warning(
                        f"Unsupported tool action during retry precheck: action={action}, tool_call={tool_call}"
                    )
            return valid_tool_calls

        for attempt in range(EMPTY_TOOL_CALL_RETRY_TIMES):
            response_message = self.call_llm(payload, model)
            response_message = self._inject_terminal_tool_call_from_content(response_message)
            tool_calls = response_message.get("tool_calls") or []
            valid_tool_calls = extract_supported_tool_calls(tool_calls)

            if valid_tool_calls:
                if attempt > 0:
                    logger.info(
                        f"Received supported tool_calls after retry {attempt + 1}/{EMPTY_TOOL_CALL_RETRY_TIMES}"
                    )
                if len(valid_tool_calls) != len(tool_calls):
                    response_message = {
                        **response_message,
                        "tool_calls": valid_tool_calls,
                    }
                return response_message

            # tool_calls 为空，或 action 不支持，先尝试 fallback 从 content 里解析, 注: 这一步理论上没用！！！！
            content = response_message.get("content") or ""
            if content:
                fallback_tool_calls, clean_content = parse_tool_calls_from_content(content)
                if fallback_tool_calls:
                    valid_fallback_tool_calls = extract_supported_tool_calls(fallback_tool_calls)
                    if valid_fallback_tool_calls:
                        logger.info(
                            f"Fallback parser recovered {len(valid_fallback_tool_calls)} supported tool_call(s) from content on attempt {attempt + 1}/{EMPTY_TOOL_CALL_RETRY_TIMES}"
                        )
                        response_message = {
                            **response_message,
                            "content": clean_content,
                            "tool_calls": valid_fallback_tool_calls,
                        }
                        return response_message
                    logger.info(
                        f"Fallback parser recovered tool_call(s), but none were supported on attempt {attempt + 1}/{EMPTY_TOOL_CALL_RETRY_TIMES}"
                    )

            # Log 检索此部分
            logger.warning(
                f"LLM response missing supported tool_calls on attempt {attempt + 1}/{EMPTY_TOOL_CALL_RETRY_TIMES}: {response_message}"
            )

            if attempt < EMPTY_TOOL_CALL_RETRY_TIMES - 1:
                time.sleep(1)

        return response_message

    # @backoff.on_exception(
    #     backoff.constant,
    #     (
    #         SSLError,
    #         openai.RateLimitError,
    #         openai.BadRequestError,
    #         openai.InternalServerError,
    #     ),
    #     interval=30,
    #     max_tries=1,
    # )
    def call_llm(self, payload, model) -> dict:
        messages = self._sanitize_messages_for_llm(payload["messages"])

        for _ in range(MAX_RETRY_TIMES):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    # top_p=self.top_p,
                    tools=json.loads(QWEN3VL_COMPUTER_USE_TOOL_SCHEMA) if self.enable_code_tool else json.loads(QWEN3VL_COMPUTER_USE_TOOL_SCHEMA_WITHOUT_CODE),
                    tool_choice="auto", # required 的话只会输出 tool_call, auto 可以自由一点
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": self.use_thinking},
                        # "repetition_penalty": 1.2, # 给予适当的重复惩罚
                    }
                )

                message_dict = response.choices[0].message.model_dump(exclude_none=True)
                return message_dict
            except Exception as e:
                logger.error(f"Error calling Qwen model: {e}")
                self.debug_print_messages()
                time.sleep(5)
                continue
        return {}

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.qwen3vl_agent")
        self.last_code_result = None
        self.messages = []
        self.pending_tool_calls = []

    def debug_print_messages(self):
        """优雅地打印 messages 列表，自动截断 Base64 图片以方便检查装填逻辑。"""
        print("\n" + "=" * 50 + " MESSAGES LOGIC DEBUG " + "=" * 50)

        if not self.messages:
            print(" [!] Messages list is empty.")
            print("=" * 122 + "\n")
            return

        for i, msg in enumerate(self.messages):
            role = msg.get("role", "UNKNOWN").upper()
            prefix = "👤" if role == "USER" else "🤖" if role == "ASSISTANT" else "⚙️" if role == "SYSTEM" else "🛠️" if role == "TOOL" else "❓" 

            print(f"\n{prefix} [{i}] ROLE: {role}")
            print("-" * 80)

            reasoning_content = msg.get("reasoning_content", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if reasoning_content:
                print(f"  (Think) : {reasoning_content}")

            if isinstance(content, str):
                print(f"  (Text) : {content}")

            for _, tool_call in enumerate(tool_calls):
                print(f"  (Tool Call): {tool_call}")
            
            if isinstance(content, list):
                for _, item in enumerate(content):
                    item_type = item.get("type", "unknown")

                    if item_type == "text":
                        text_content = item.get("text", "")
                        if "Code Execution Result:" in text_content:
                            print(f"  📝 (Text-CodeResult) : {text_content}")
                        else:
                            if len(text_content) > 2000:
                                text_content = text_content[:2000] + " ... [TEXT TRUNCATED]"
                            print(f"  💬 (Text)  : {text_content}")

                    elif item_type == "image_url":
                        print(f"  🖼️ (Image) : <BASE64_IMAGE_DATA_TRUNCATED>")
                    else:
                        print(f"  ❓ ({item_type}) : {item}")

        print("\n" + "=" * 122 + "\n")
