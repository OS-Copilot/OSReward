import base64
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap
import logging

def round_by_factor(number: int, factor: int) -> int:
    """返回最接近 number 的且能被 factor 整除的整数"""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """返回大于等于 number 的且能被 factor 整除的整数"""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """返回小于等于 number 的且能被 factor 整除的整数"""
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280, max_long_side=8192):
    """缩放后图片满足以下条件:
    1. 长宽能被 factor 整除
    2. pixels 总数被限制在 [min_pixels, max_pixels] 内
    3. 最长边限制在 max_long_side 内
    4. 保证其长宽比基本不变
    """
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 100, got {height} / {width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def update_image_size_(image_ele: dict, min_tokens=1, max_tokens=12800, merge_base=2, patch_size=14):
    """根据 min_tokens, max_tokens 更新 image_ele 的尺寸信息

    Args:
        image_ele (dict):
            - image_ele["image"]: str 图片路径
            - image_ele["height"]: int 图片原始高度
            - image_ele["width"]: int 图片原始宽度

    Returns:
        更新后的 image_ele, 新增如下 key-value pair
        dict:
            - image_ele["resized_height"]: int 输入到模型的真实高度
            - image_ele["resized_width"]: int 输入到模型的真实宽度
            - image_ele["seq_len"]: int 输入到模型所占的序列长度
    """
    height, width = image_ele["height"], image_ele["width"]
    pixels_per_token = patch_size * patch_size * merge_base * merge_base
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=merge_base * patch_size,
        min_pixels=pixels_per_token * min_tokens,
        max_pixels=pixels_per_token * max_tokens,
        max_long_side=50000,
    )
    image_ele.update(
        {
            "resized_height": resized_height,
            "resized_width": resized_width,
            "seq_len": resized_height * resized_width // pixels_per_token + 2,
        }
    )
    return image_ele


def _convert_bbox_format_from_abs_origin(bbox, image_ele: dict, *, tgt_format: str):
    x1, y1, x2, y2 = bbox
    if tgt_format == "abs_origin":
        new_bbox = [int(x1), int(y1), int(x2), int(y2)]
    elif tgt_format == "abs_resized":
        new_bbox = [
            int(x1 / image_ele["width"] * image_ele["resized_width"]),
            int(y1 / image_ele["height"] * image_ele["resized_height"]),
            int(x2 / image_ele["width"] * image_ele["resized_width"]),
            int(y2 / image_ele["height"] * image_ele["resized_height"]),
        ]
    elif tgt_format == "qwen-vl":
        new_bbox = [
            int(x1 / image_ele["width"] * 999),
            int(y1 / image_ele["height"] * 999),
            int(x2 / image_ele["width"] * 999),
            int(y2 / image_ele["height"] * 999),
        ]
    elif tgt_format == "rel":
        new_bbox = [
            float(x1 / image_ele["width"]),
            float(y1 / image_ele["height"]),
            float(x2 / image_ele["width"]),
            float(y2 / image_ele["height"]),
        ]
    elif tgt_format == "molmo":
        new_bbox = [
            round(x1 / image_ele["width"] * 100, ndigits=1),
            round(y1 / image_ele["height"] * 100, ndigits=1),
            round(x2 / image_ele["width"] * 100, ndigits=1),
            round(y2 / image_ele["height"] * 100, ndigits=1),
        ]
    else:
        assert False, f"Unknown tgt_format: {tgt_format}"
    return new_bbox


def _convert_bbox_format_to_abs_origin(bbox, image_ele: dict, *, src_format: str):
    x1, y1, x2, y2 = bbox
    if src_format == "abs_origin":
        new_bbox = [int(x1), int(y1), int(x2), int(y2)]
    elif src_format == "abs_resized":
        new_bbox = [
            int(x1 / image_ele["resized_width"] * image_ele["width"]),
            int(y1 / image_ele["resized_height"] * image_ele["height"]),
            int(x2 / image_ele["resized_width"] * image_ele["width"]),
            int(y2 / image_ele["resized_height"] * image_ele["height"]),
        ]
    elif src_format == "qwen-vl":
        new_bbox = [
            int(x1 / 999 * image_ele["width"]),
            int(y1 / 999 * image_ele["height"]),
            int(x2 / 999 * image_ele["width"]),
            int(y2 / 999 * image_ele["height"]),
        ]
    elif src_format == "rel":
        new_bbox = [
            int(x1 * image_ele["width"]),
            int(y1 * image_ele["height"]),
            int(x2 * image_ele["width"]),
            int(y2 * image_ele["height"]),
        ]
    elif src_format == "molmo":
        new_bbox = [
            int(x1 / 100 * image_ele["width"]),
            int(y1 / 100 * image_ele["height"]),
            int(x2 / 100 * image_ele["width"]),
            int(y2 / 100 * image_ele["height"]),
        ]
    else:
        assert False, f"Unknown src_format: {src_format}"
    return new_bbox


def convert_bbox_format(bbox, image_ele: dict, *, src_format: str, tgt_format: str):
    bbox_abs_origin = _convert_bbox_format_to_abs_origin(bbox, image_ele, src_format=src_format)
    bbox_tgt_format = _convert_bbox_format_from_abs_origin(bbox_abs_origin, image_ele, tgt_format=tgt_format)
    return bbox_tgt_format


def _convert_point_format_from_abs_origin(point, image_ele: dict, *, tgt_format: str):
    x, y = point
    if tgt_format == "abs_origin":
        new_point = [int(x), int(y)]
    elif tgt_format == "abs_resized":
        new_point = [
            int(x / image_ele["width"] * image_ele["resized_width"]),
            int(y / image_ele["height"] * image_ele["resized_height"]),
        ]
    elif tgt_format == "qwen-vl":
        new_point = [
            int(x / image_ele["width"] * 999),
            int(y / image_ele["height"] * 999),
        ]
    elif tgt_format == "rel":
        new_point = [
            float(x / image_ele["width"]),
            float(y / image_ele["height"]),
        ]
    elif tgt_format == "molmo":
        new_point = [
            round(x / image_ele["width"] * 100, ndigits=1),
            round(y / image_ele["height"] * 100, ndigits=1),
        ]
    else:
        assert False, f"Unknown tgt_format: {tgt_format}"
    return new_point


def _convert_point_format_to_abs_origin(point, image_ele: dict, *, src_format: str):
    x, y = point
    if src_format == "abs_origin":
        new_point = [int(x), int(y)]
    elif src_format == "abs_resized":
        new_point = [
            int(x / image_ele["resized_width"] * image_ele["width"]),
            int(y / image_ele["resized_height"] * image_ele["height"]),
        ]
    elif src_format == "qwen-vl":
        new_point = [
            int(x / 999 * image_ele["width"]),
            int(y / 999 * image_ele["height"]),
        ]
    elif src_format == "rel":
        new_point = [
            int(x * image_ele["width"]),
            int(y * image_ele["height"]),
        ]
    elif src_format == "molmo":
        new_point = [
            int(x / 100 * image_ele["width"]),
            int(y / 100 * image_ele["height"]),
        ]
    else:
        assert False, f"Unknown src_format: {src_format}"
    return new_point


def convert_point_format(point, image_ele: dict, *, src_format: str, tgt_format: str):
    point_abs_origin = _convert_point_format_to_abs_origin(point, image_ele, src_format=src_format)
    point_tgt_format = _convert_point_format_from_abs_origin(point_abs_origin, image_ele, tgt_format=tgt_format)
    return point_tgt_format


__all__ = [
    "update_image_size_",
    "convert_bbox_format",
    "convert_point_format",
]

def dedup_and_save_images_for_gemini(                     
    messages: list,                                                                                                                                                                  
    image_hash_map: dict[str, str],
    image_root_dir: Path,                                                                                                                                                            
) -> tuple[list[str], dict[str, str]]:                    
    """去重保存 Gemini 轨迹中的图像到 Qwen3VL 目录，返回当前样本使用到的文件名列表。
                                                                                                                                                                                    
    image_hash_map: base64_sha1 -> filename，用于跨 step 复用相同图像文件。                                                                                                          
                                                                                                                                                                                    
    Gemini 的截图结构为：                                                                                                                                                            
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64...>"}}
                                                                                                                                                                                    
    注意：                                                
    - 我们扫描所有 message 的顶层 content；                                                                                                                                          
    - 只认 type == "image_url" 且 image_url.url 是 data:...;base64,... 形式；                                                                                                        
    - 每发现一张图，就按 base64 内容去重并写入 image_root_dir。                                                                                                                      
    """                                                                                                                                                                              
    used_filenames: list[str] = []                                                                                                                                                   
                                                                                                                                                                                    
    def _handle_single_image_block(img_block: dict):      
        nonlocal image_hash_map, used_filenames
                                                                                                                                                                                    
        if not isinstance(img_block, dict):
            return                                                                                                                                                                   
        if img_block.get("type") != "image_url":          
            return

        image_url = img_block.get("image_url") or {}                                                                                                                                 
        if not isinstance(image_url, dict):
            return                                                                                                                                                                   
                                                        
        url: str = image_url.get("url") or ""                                                                                                                                        
        if not url:
            return                                                                                                                                                                   
                                                        
        # 只处理 data URL: data:image/png;base64,<BASE64...>                                                                                                                         
        if not url.startswith("data:"):
            return                                                                                                                                                                   
                                                        
        try:                                                                                                                                                                         
            header, b64_data = url.split(",", 1)          
        except ValueError:                                                                                                                                                           
            return                                                                                                                                                                   
                                                                                                                                                                                    
        media_type = "image/png"                                                                                                                                                     
        if header.startswith("data:") and ";base64" in header:
            # 例如 data:image/png;base64
            media_type = header.split(";", 1)[0].removeprefix("data:") or "image/png"                                                                                                

        data = b64_data.strip()                                                                                                                                                      
        if not data:                                      
            return
                                                                                                                                                                                    
        # 对 base64 字符串做 sha1
        h = hashlib.sha1(data.encode("utf-8")).hexdigest()                                                                                                                           
        if h in image_hash_map:                           
            filename = image_hash_map[h]                                                                                                                                             
        else:
            ext = "png" if "png" in media_type else "jpg"                                                                                                                            
            filename = f"{h}.{ext}"                                                                                                                                                  
            out_path = image_root_dir / filename
            if not out_path.exists():                                                                                                                                                
                try:                                      
                    raw_bytes = base64.b64decode(data)
                    out_path.write_bytes(raw_bytes)
                except Exception:
                    # 单条图像失败不影响整体样本
                    return                                                                                                                                                           
            image_hash_map[h] = filename
                                                                                                                                                                                    
        used_filenames.append(filename)

    # 遍历所有 messages 顶层 content，查找 image_url                                                                                                                                 
    for m in messages:
        if not isinstance(m, dict):                                                                                                                                                  
            continue                                      
        content = m.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            _handle_single_image_block(block)
                                                                                                                                                                                    
    return used_filenames, image_hash_map

def dedup_and_save_images_for_claude(
    messages: list,
    image_hash_map: dict[str, str],
    image_root_dir: Path
) -> tuple[list[str], dict[str, str]]:
    """去重保存多模态图像到 Qwen3VL 目录，返回当前样本使用到的文件名列表。

    image_hash_map: base64_sha1 -> filename，用于跨 step 复用相同图像文件。

    注意：
    - Claude 的截图主要位于 tool_result block 的 content 里；
    - 预处理后首屏截图会被放到第一个 user 的顶层 image block 中；
    因此这里需要同时扫描 user 顶层 image 和 tool_result 内部的 image，保证 images 列表与 <image> 占位一一对应。
    """
    used_filenames: list[str] = []

    def _handle_single_image_block(img_block: dict):
        nonlocal image_hash_map, used_filenames
        src = img_block.get("source") or {}
        if not (isinstance(src, dict) and src.get("type") == "base64"):
            return
        data = src.get("data")
        media_type = src.get("media_type", "image/png")
        if not data:
            return

        # 直接对 base64 字符串做 sha1，避免重复解码
        h = hashlib.sha1(data.encode("utf-8")).hexdigest()
        if h in image_hash_map:
            filename = image_hash_map[h]
        else:
            ext = "png" if "png" in media_type else "jpg"
            filename = f"{h}.{ext}"
            out_path = image_root_dir / filename
            if not out_path.exists():
                try:
                    raw_bytes = base64.b64decode(data)
                    out_path.write_bytes(raw_bytes)
                except Exception:
                    # 单条图像失败不影响整体样本
                    return
            image_hash_map[h] = filename

        used_filenames.append(filename)

    # 遍历 messages：
    # 1) user 顶层 image（例如首屏截图）
    # 2) tool_result.content 中的 image（工具返回截图）
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue

        # 1) 顶层 image（多出现在 user 首条）
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                _handle_single_image_block(block)

        # 2) tool_result 内部 image
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            sub_content = block.get("content")
            if not isinstance(sub_content, list):
                continue
            for sub in sub_content:
                if isinstance(sub, dict) and sub.get("type") == "image":
                    _handle_single_image_block(sub)

    return used_filenames, image_hash_map

QWEN3VL_COMPUTER_USE_TOOL_SCHEMA = json.dumps(
[{
        "type": "function",
        "function": {
            "name": "custom_computer_use",
            "description": (                                            
                "Control a desktop GUI and execute system-level code."
                "Use it to move the mouse, click, type, scroll, wait, terminate tasks,"
                "and run raw Python or Bash code on the operating system."
            ),
            "parameters": {
                "properties": {
                    "action": {
                        "description": textwrap.dedent("""
                        The type of operation to perform: 
                        * `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
                        * `type`: Type a string of text on the keyboard.
                        * `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
                        * `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
                        * `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
                        * `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
                        * `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
                        * `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
                        * `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
                        * `scroll`: Performs a vertical scroll of the mouse scroll wheel.
                        * `hscroll`: Performs a horizontal scroll of the mouse scroll wheel.
                        * `wait`: Wait specified seconds for the change to happen.
                        * `terminate`: Terminate the current task and report its completion status.
                        * `code`: Execute raw Python or Bash scripts to perform tasks directly in the operating system.
                        """),
                        "type": "string",
                        "enum": [
                            "key", "type", "mouse_move", "left_click", "left_click_drag",
                            "right_click", "middle_click", "double_click", "triple_click", "scroll", "hscroll",
                            "wait", "terminate", "code"
                        ],
                    },
                    "keys": {"description": "Required only by `action=key`.", "type": "array"},
                    "text": {"description": "Required only by `action=type`.", "type": "string"},
                    "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                    "pixels": {"description": "The amount of scrolling.", "type": "number"},
                    "time": {"description": "The seconds to wait.", "type": "number"},
                    "status": {
                        "description": "The status of the task.", 
                        "type": "string", 
                        "enum": ["success", "failure"]
                    },
                    "execute_code": {
                        "description": "The raw code string to execute. Required only when `action=code`.",
                        "type": "string"
                    },
                    "language": {
                        "description": "The programming language of the code. Required only when `action=code`.",
                        "type": "string",
                        "enum": ["python", "bash"]
                    }
                },
                "type": "object",
                "required": ["action"],
            },
        },
    }],
    ensure_ascii=False,
)

QWEN3VL_COMPUTER_USE_TOOL_SCHEMA_WITHOUT_CODE = json.dumps(
[
  {
    "type": "function",
    "function": {
      "name": "computer_use",
      "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is 1000x1000.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
      "parameters": {
        "type": "object",
        "required": ["action"],
        "properties": {
          "action": {
            "type": "string",
            "enum": [
              "key",
              "type",
              "mouse_move",
              "left_click",
              "left_click_drag",
              "right_click",
              "middle_click",
              "double_click",
              "triple_click",
              "hscroll",
              "scroll",
              "wait",
              "terminate"
            ],
            "description": textwrap.dedent("""
                The type of operation to perform: 
                * `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
                * `type`: Type a string of text on the keyboard.
                * `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
                * `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
                * `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
                * `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
                * `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
                * `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
                * `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
                * `scroll`: Performs a vertical scroll of the mouse scroll wheel.
                * `hscroll`: Performs a horizontal scroll of the mouse scroll wheel.
                * `wait`: Wait specified seconds for the change to happen.
                * `terminate`: Terminate the current task and report its completion status.
                """),          
            },
            "keys": {
                "type": "array",
                "description": "Required only by `action=key`."
            },
            "text": {
                "type": "string",
                "description": "Required only by `action=type`."
            },
            "coordinate": {
                "type": "array",
                "description": "The x,y coordinates for mouse actions."
            },
            "pixels": {
                "type": "number",
                "description": "The amount of scrolling."
            },
            "time": {
                "type": "number",
                "description": "The seconds to wait."
            },
            "status": {
                "type": "string",
                "enum": ["success", "failure"],
                "description": "The status of the task."
            }
        }
      }
    }
  }
],
ensure_ascii=False,
)

QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_FOR_TRAIN = textwrap.dedent("""
# Role & Goal
You are a powerful OS Agent capable of both GUI interaction and direct system-level programming and are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
Your goal is to complete tasks with MAXIMUM efficiency and MINIMUM steps.

# Environment & Screen
- The user's home directory is "/home/user".
- The user's sudo password is "password".
- The screen's resolution is represented on a 1000x1000 relative coordinate grid.
""")

QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_ADDITIONAL_RULES = textwrap.dedent("""
# Additional Rules & Action Guidelines

### 1. Action Selection Strategy
**Prioritize `code` actions for:**
- **Data Processing:** Parsing or manipulating structured data (e.g., CSV, Excel, JSON).
- **Precision Tasks**: Executing tasks that would otherwise require high-precision GUI interactions (which are prone to OCR and spatial reasoning failures).
- **Batch Operations:** Bulk file management (rename, copy, move, delete).
- **Text Manipulation:** Complex search/replace across files or within large documents.

**Reserve GUI actions for:**
- **System Navigation:** Launching, focusing, or switching between applications.
- **Basic UI Interaction:** Interacting with large, prominent application controls (e.g., standard menus, distinct buttons) where pixel-perfect precision is NOT required.
- **Non-Programmable Tasks:** Navigating browsers or desktop applications where no CLI/API is readily available.

### 2. Code Execution & Verification Workflow
- **Pre-execution File Location:** Before executing any `code` to process or modify a file, you MUST first locate the target file within the **user's home directory**.
- **In-Place Modification Default:** Unless explicitly instructed to create a new file, a new sheet, or a copy, you MUST modify the target file in-place. Do not alter the original filename, and strictly preserve all pre-existing content, formats, or structural elements (e.g., untouched columns, rows, or other sheets) that are not targeted by the user's instruction.
- **Evaluate Output:** Immediately after executing a `code` action, analyze the textual output (stdout/stderr) to assess success before taking the next step.
- **Rigorous Content Verification:** Because code executes in the background, you MUST explicitly verify that the modifications were successfully saved and are reflected correctly. Examples of effective verification include (but are not limited to):
    1. **GUI Reopen:** Use GUI actions to close the file (do NOT save during closing) and reopen it.
    2. **Shortcut Reopen:** Send the `ctrl w` shortcut to close the active file/tab, then reopen it.
    3. **Code Print:** Execute a secondary `code` action to print the modified file's contents to the terminal (e.g., using `cat`, `head`, or a simple Python script).
- **GUI Fallback:** If code-based approaches fail or encounter persistent errors, gracefully pivot to using GUI actions to complete the task.
- **Avoid Timeout**: If you need to launch GUI applications or persistent background processes, you MUST fully detach them from the parent process's output pipes to prevent blocking. Always use the format `nohup <command> > /dev/null 2>&1 &` to ensure the script returns immediately without hitting the timeout.
                                                                                                                         
### 3. Environment & Dependencies
- **Pre-installed Packages:** You have direct access to `ffmpeg`, `ffmpeg-python`, `av`, `python-pptx`, `python-docx`, `openpyxl`, `pillow`, `pydub`, `PyMuPDF`, `pdfplumber`, etc.
- **Dynamic Installation:** You are authorized to install any missing dependencies as needed to accomplish the task.

# Output Contract
Before each tool call, briefly explain your reasoning: why this action is needed and what you expect to happen.

Do NOT skip this reasoning block, and do NOT call the tool without it appearing immediately above.
""")

QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_FOR_OSWORLD_INFERENCE = QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_FOR_TRAIN + QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_ADDITIONAL_RULES

QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_FOR_WAA_INFERENCE = textwrap.dedent("""
# Role & Goal
You are a powerful OS Agent capable of both GUI interaction and direct system-level programming and are utilising a Windows 11 virtual machine using x86_64 architecture with internet access.
Your goal is to complete tasks with MAXIMUM efficiency and MINIMUM steps.

# Environment & Screen
- **User**: Your username is "Docker".
- **Home**: Your home path is "C:\\Users\\Docker"
- The screen's resolution is represented on a 1000x1000 relative coordinate grid.
""") + QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_ADDITIONAL_RULES
 
QWEN3VL_COMPUTER_USE_SYSTEM_PROMPT_FOR_OSWORLD_INFERENCE_WITHOUT_CODE = textwrap.dedent("""
# Role & Goal
You are a powerful GUI Agent and are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
Your goal is to complete tasks with MAXIMUM efficiency and MINIMUM steps.

# Environment & Screen
- The user's home directory is "/home/user".
- The user's sudo password is "password".
- The screen's resolution is represented on a 1000x1000 relative coordinate grid.

# Output Contract
Before each tool call, briefly explain your reasoning: why this action is needed and what you expect to happen.

Do NOT skip this reasoning block, and do NOT call the tool without it appearing immediately above.
""")

KIMI_COMPUTER_USE_SYSTEM_PROMPT_FOR_OSWORLD_INFERENCE = textwrap.dedent("""
# Role & Goal
You are a powerful OS Agent capable of both GUI interaction and direct system-level programming and are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
Your goal is to complete tasks with MAXIMUM efficiency and MINIMUM steps.

# Environment & Screen
- The user's home directory is "/home/user".
- The user's sudo password is "password".

# Additional Rules & Action Guidelines

### 1. Action Selection Strategy
**Prioritize `code` actions for:**
- **Data Processing:** Parsing or manipulating structured data (e.g., CSV, Excel, JSON).
- **Precision Tasks**: Executing tasks that would otherwise require high-precision GUI interactions (which are prone to OCR and spatial reasoning failures).
- **Batch Operations:** Bulk file management (rename, copy, move, delete).
- **Text Manipulation:** Complex search/replace across files or within large documents.

**Reserve GUI actions for:**
- **System Navigation:** Launching, focusing, or switching between applications.
- **Basic UI Interaction:** Interacting with large, prominent application controls (e.g., standard menus, distinct buttons) where pixel-perfect precision is NOT required.
- **Non-Programmable Tasks:** Navigating browsers or desktop applications where no CLI/API is readily available.

### 2. Code Execution & Verification Workflow
- **Pre-execution File Location:** Before executing any `code` to process or modify a file, you MUST first locate the target file within the **user's home directory**.
- **In-Place Modification Default:** Unless explicitly instructed to create a new file, a new sheet, or a copy, you MUST modify the target file in-place. Do not alter the original filename, and strictly preserve all pre-existing content, formats, or structural elements (e.g., untouched columns, rows, or other sheets) that are not targeted by the user's instruction.
- **Evaluate Output:** Immediately after executing a `code` action, analyze the textual output (stdout/stderr) to assess success before taking the next step.
- **Rigorous Content Verification:** Because code executes in the background, you MUST explicitly verify that the modifications were successfully saved and are reflected correctly. Examples of effective verification include (but are not limited to):
    1. **GUI Reopen:** Use GUI actions to close the file (do NOT save during closing) and reopen it.
    2. **Shortcut Reopen:** Send the `ctrl w` shortcut to close the active file/tab, then reopen it.
    3. **Code Print:** Execute a secondary `code` action to print the modified file's contents to the terminal (e.g., using `cat`, `head`, or a simple Python script).
- **GUI Fallback:** If code-based approaches fail or encounter persistent errors, gracefully pivot to using GUI actions to complete the task.
- **Avoid Timeout**: If you need to launch GUI applications or persistent background processes, you MUST fully detach them from the parent process's output pipes to prevent blocking. Always use the format `nohup <command> > /dev/null 2>&1 &` to ensure the script returns immediately without hitting the timeout.
                                                                                                                         
### 3. Environment & Dependencies
- **Pre-installed Packages:** You have direct access to `ffmpeg`, `ffmpeg-python`, `av`, `python-pptx`, `python-docx`, `openpyxl`, `pillow`, `pydub`, `PyMuPDF`, `pdfplumber`, etc.
- **Dynamic Installation:** You are authorized to install any missing dependencies as needed to accomplish the task.

### 4. Action Space Notes
1. **scroll / hscroll:** Must include a `coordinate` parameter to specify the scroll position, and a `pixels` parameter to indicate the number of scroll wheel clicks (positive or negative), which maps to the `clicks` argument in `pyautogui.scroll/hscroll(clicks=xx)`.
2. **drag:** A drag operation typically requires two actions in combination: `mouse_move` followed by `left_click_drag` to complete the full drag behavior.
3. All output coordinates must be absolute coordinates. Most importantly, DON'T explicitly mention specific coordinates (e.g. x=300) in any text outside of the tool_call.
""")
