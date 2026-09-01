from __future__ import annotations
import base64
import inspect
import logging
import os
import time
import re
from typing import Callable, Any, Optional, Tuple
from typing import List, Dict, Union
import ast
import gymnasium as gym

from desktop_env.controllers.python import PythonController
from desktop_env.controllers.setup import SetupController
from desktop_env.evaluators import metrics, getters
from desktop_env.evaluators.scalecua import (
    resolve_getter as resolve_scalecua_getter,
    resolve_metric as resolve_scalecua_metric,
)
from desktop_env.providers import create_vm_manager_and_provider

logger = logging.getLogger("desktopenv.env")

Metric = Callable[[Any, Any], float]
Getter = Callable[[gym.Env, Dict[str, Any]], Any]

MAX_RETRIES = 5 # Maximum retries for environment setup


class EnvironmentSetupError(RuntimeError):
    """Raised when task setup fails after retries."""
            


def _fix_pyautogui_less_than_bug(command: str) -> str:
    """
    Fix PyAutoGUI '<' character bug by converting it to hotkey("shift", ',') calls.
    
    This fixes the known PyAutoGUI issue where typing '<' produces '>' instead.
    References:
    - https://github.com/asweigart/pyautogui/issues/198
    - https://github.com/xlang-ai/OSWorld/issues/257
    
    Args:
        command (str): The original pyautogui command
        
    Returns:
        str: The fixed command with '<' characters handled properly
    """
    # Pattern to match press('<') or press('\u003c') calls  
    press_pattern = r'pyautogui\.press\(["\'](?:<|\\u003c)["\']\)'

    # Handle press('<') calls
    def replace_press_less_than(match):
        return 'pyautogui.hotkey("shift", ",")'
    
    # First handle press('<') calls
    command = re.sub(press_pattern, replace_press_less_than, command)

    # Pattern to match typewrite calls with quoted strings
    typewrite_pattern = r'pyautogui\.write\((["\'])(.*?)\1\)'
    
    # Then handle typewrite calls
    def process_typewrite_match(match):
        quote_char = match.group(1)
        content = match.group(2)
        
        # Preprocess: Try to decode Unicode escapes like \u003c to actual '<'
        # This handles cases where '<' is represented as escaped Unicode
        try:
            # Attempt to decode unicode escapes
            decoded_content = content.encode('utf-8').decode('unicode_escape')
            content = decoded_content
        except UnicodeDecodeError:
            # If decoding fails, proceed with original content to avoid breaking existing logic
            pass  # English comment: Graceful degradation - fall back to original content if decoding fails
        
        # Check if content contains '<'
        if '<' not in content:
            return match.group(0)
        
        # Split by '<' and rebuild
        parts = content.split('<')
        result_parts = []
        
        for i, part in enumerate(parts):
            if i == 0:
                # First part
                if part:
                    result_parts.append(f"pyautogui.write({quote_char}{part}{quote_char})")
            else:
                # Add hotkey for '<' and then typewrite for the rest
                result_parts.append('pyautogui.hotkey("shift", ",")')
                if part:
                    result_parts.append(f"pyautogui.write({quote_char}{part}{quote_char})")
        
        return '; '.join(result_parts)
    
    command = re.sub(typewrite_pattern, process_typewrite_match, command)
    
    return command


class DesktopEnv(gym.Env):
    """
    DesktopEnv with OpenAI Gym interface. It provides a desktop environment for setting and evaluating desktop automation tasks.
    """
    def __init__(
            self,
            provider_name: str = "vmware",
            region: str = None,
            path_to_vm: str = None,
            snapshot_name: str = "init_state",
            action_space: str = "pyautogui",
            cache_dir: str = "cache",
            screen_size: Tuple[int] = (int(os.environ.get("SCREEN_WIDTH", 1920)), int(os.environ.get("SCREEN_HEIGHT", 1080))),
            headless: bool = False,
            require_a11y_tree: bool = True,
            require_terminal: bool = False,
            os_type: str = "Ubuntu",
            enable_proxy: bool = False,
            client_password: str = "",
            proxy: str = ""
    ):
        """
        Args:
            provider_name (str): virtualization provider name, default to "vmware"
            region (str): the region for allocate machines, work for cloud services, default to  "us-east-1"
            path_to_vm (str): path to .vmx file
            snapshot_name (str): snapshot name to revert to, default to "init_state"
            action_space (str): "computer_13" | "pyautogui"
            cache_dir (str): cache directory to cache task-related stuffs like
              reference file for evaluation
            screen_size (Tuple[int]): screen size of the VM
            headless (bool): whether to run the VM in headless mode
            require_a11y_tree (bool): whether to require accessibility tree
            require_terminal (bool): whether to require terminal output
            os_type (str): operating system type, default to "Ubuntu"
            enable_proxy (bool): whether to enable proxy support, default to False
        """
        # Initialize VM manager and vitualization provider
        self.region = region
        self.provider_name = provider_name
        self.enable_proxy = enable_proxy  # Store proxy enablement setting
        if client_password == "":
            if self.provider_name == "aws":
                self.client_password = "osworld-public-evaluation"
            else:
                self.client_password = "password"
        else:
            self.client_password = client_password

        self.screen_width = screen_size[0]
        self.screen_height = screen_size[1]

        # Default 
        self.server_port = 5000
        self.chromium_port = 9222
        self.vnc_port = 8006
        self.vlc_port = 8080
        self.proxy = proxy
        
        # Initialize with default (no proxy) provider
        self.current_use_proxy = False
        self.manager, self.provider = None, None

        self.os_type = os_type

        # Track whether environment has been used (step/setup) to optimize snapshot revert
        # docker and aws are always unused as the emulator starts from a clean state
        # vmware is always used as the emulator starts from a dirty state
        if self.provider_name in {"docker", "aws"}:
            self.is_environment_used = False
        elif self.provider_name == "vmware":
            self.is_environment_used = True
        else:
            raise ValueError(f"Invalid provider name: {self.provider_name}")

        # Initialize environment variables
        self.path_to_vm = path_to_vm
        # if path_to_vm:
        #     self.path_to_vm = os.path.abspath(os.path.expandvars(os.path.expanduser(path_to_vm))) \
        #         if provider_name in {"vmware", "virtualbox"} else path_to_vm
        # else:
        #     self.path_to_vm = self.manager.get_vm_path(os_type=self.os_type, region=region, screen_size=(self.screen_width, self.screen_height))
        
        self.snapshot_name = snapshot_name
        self.cache_dir_base: str = cache_dir
        # todo: add the logic to get the screen size from the VM
        self.headless = headless
        self.require_a11y_tree = require_a11y_tree
        self.require_terminal = require_terminal

        # Initialize emulator and controller
        # logger.info("Initializing...")
        # self._start_emulator() # Start the emulator, we should seperate it with another method

        # mode: human or machine
        self.instruction = None
        assert action_space in ["computer_13", "pyautogui", "claude_computer_use", "autoglm_computer_use"]
        self.action_space = action_space  # todo: refactor it to the ActType

        # episodic stuffs, like counters, will be updated or reset
        # when calling self.reset()
        self._traj_no: int = -1
        self._step_no: int = 0
        self.action_history: List[Dict[str, any]] = []

        self.metric = None

    # 将物理上的初始化和逻辑上的初始化解耦，便于传参
    def start(self):
        # Initialize emulator and controller
        if not self.manager and not self.provider:
            logger.info("Initializing...")
            self.manager, self.provider = create_vm_manager_and_provider(self.provider_name, self.region, use_proxy=False)
            if self.path_to_vm:
                self.path_to_vm = os.path.abspath(os.path.expandvars(os.path.expanduser(self.path_to_vm))) \
                    if self.provider_name == "vmware" else self.path_to_vm
            else:
                self.path_to_vm = self.manager.get_vm_path(os_type=self.os_type, region=self.region, screen_size=(self.screen_width, self.screen_height))
            self._start_emulator()

    def _start_emulator(self):
        assert self.manager and self.provider and self.path_to_vm
        try:
            # Power on the virtual machine
            self.provider.start_emulator(self.path_to_vm, self.headless, self.os_type)

            # Get the ip from the virtual machine, and setup the controller
            vm_ip_ports = self.provider.get_ip_address(self.path_to_vm).split(':')
            self.vm_ip = vm_ip_ports[0]
            # Get the ports from the virtual machine (for Docker provider only)
            if len(vm_ip_ports) > 1:
                self.server_port = int(vm_ip_ports[1])
                self.chromium_port = int(vm_ip_ports[2])
                self.vnc_port = int(vm_ip_ports[3])
                self.vlc_port = int(vm_ip_ports[4])
            self.controller = PythonController(vm_ip=self.vm_ip, server_port=self.server_port, width=self.screen_width, height=self.screen_height)
            self.setup_controller = SetupController(vm_ip=self.vm_ip, server_port=self.server_port, chromium_port=self.chromium_port, vlc_port=self.vlc_port, cache_dir=self.cache_dir_base, client_password=self.client_password, screen_width=self.screen_width, screen_height=self.screen_height)
            self.v2_setup_controller = None

        except Exception as e:
            try:
                self.provider.stop_emulator(self.path_to_vm)
            except Exception as stop_err:
                logger.warning(f"Cleanup after interrupt failed: {stop_err}")
            raise

    def _revert_to_snapshot(self):
        # Revert to certain snapshot of the virtual machine, and refresh the path to vm and ip of vm
        # due to the fact it could be changed when implemented by cloud services
        path_to_vm = self.provider.revert_to_snapshot(self.path_to_vm, self.snapshot_name)
        if path_to_vm and not path_to_vm == self.path_to_vm:
            # path_to_vm has to be a new path 
            
            self.manager.delete_vm(self.path_to_vm, self.region)
            self.manager.add_vm(path_to_vm, self.region)
            self.manager.occupy_vm(path_to_vm, os.getpid(), self.region)
            self.path_to_vm = path_to_vm

    def _save_state(self, snapshot_name=None):
        # Save the current virtual machine state to a certain snapshot name
        self.provider.save_state(self.path_to_vm, snapshot_name)

    def close(self):
        # Close (release) the virtual machine
        if self.provider:
            self.provider.stop_emulator(self.path_to_vm)

    @staticmethod
    def _task_get(task_config: Any, key: str, default: Any = None) -> Any:
        if task_config is None:
            return default
        if hasattr(task_config, "get") and callable(getattr(task_config, "get")):
            try:
                return task_config.get(key, default)
            except TypeError:
                return task_config.get(key)
        return getattr(task_config, key, default)

    @staticmethod
    def _has_custom_setup(task_config: Any) -> bool:
        return hasattr(task_config, "setup") and callable(getattr(task_config, "setup"))

    @staticmethod
    def _has_custom_evaluate(task_config: Any) -> bool:
        return hasattr(task_config, "evaluate") and callable(getattr(task_config, "evaluate"))

    def _get_task_setup_controller(self, task_config: Any):
        return self.setup_controller

    def _call_task_setup(self, task_config: Any, use_proxy: bool) -> None:
        setup_fn = task_config.setup
        setup_controller = self._get_task_setup_controller(task_config)
        try:
            params = inspect.signature(setup_fn).parameters
        except (TypeError, ValueError):
            setup_fn(setup_controller, use_proxy)
            return

        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in params.values())
        if "use_proxy" in params or accepts_kwargs:
            setup_fn(setup_controller, use_proxy=use_proxy)
        else:
            setup_fn(setup_controller)

    def _call_task_evaluate(self, task_config: Any):
        eval_fn = task_config.evaluate
        try:
            params = inspect.signature(eval_fn).parameters
        except (TypeError, ValueError):
            result = eval_fn(self)
        else:
            result = eval_fn(self) if params else eval_fn()
        return result if isinstance(result, dict) else float(result)

    # 每次新任务时先执行该函数
    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        
        # Reset to certain task in OSWorld
        logger.info("Resetting environment...")
        logger.info("Switching task...")
        logger.info("Setting counters...")
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()

        for attempt in range(MAX_RETRIES):
            # Only revert to snapshot if environment has been used (step/setup)
            # This optimization is especially important for cloud providers like AWS
            # where unnecessary snapshot operations are costly and time-consuming
            
            if task_config is not None:
                # Only consider task proxy requirement if proxy is enabled at system level
                task_proxy = self._task_get(task_config, "proxy", False)
                task_use_proxy = task_proxy and self.enable_proxy
                if not self.enable_proxy and task_proxy:
                    logger.info("Task requires proxy but proxy is disabled at system level, ignoring proxy requirement.")
                
                if task_use_proxy != self.current_use_proxy:
                    # keep because get_info_from_website depend on this
                    self.current_use_proxy = task_use_proxy
            
            if self.is_environment_used:
                logger.info("Environment has been used, reverting to snapshot {}...".format(self.snapshot_name))
                self._revert_to_snapshot()
                logger.info("Starting emulator...")
                self._start_emulator()
                logger.info("Emulator started.")
                # Reset the usage flag after reverting
                self.is_environment_used = False
            else:
                logger.info("Environment is clean, skipping snapshot revert (provider: {}).".format(self.provider_name))

            if task_config is not None:
                use_proxy = self._task_get(task_config, "proxy", False) and self.enable_proxy
                if use_proxy:
                    # If using proxy and proxy is enabled, set up the proxy configuration
                    # 下面这么设置代理没用
                    self.setup_controller._proxy_setup(self.client_password)

                self._set_task_info(task_config)
                self.setup_controller.reset_cache_dir(self.cache_dir)
                logger.info("Setting up environment...")
                if self._has_custom_setup(task_config):
                    try:
                        self._call_task_setup(task_config, use_proxy)
                    except Exception as e:
                        raise EnvironmentSetupError(f"Custom task setup failed: {e}") from e
                    success = True
                else:
                    success = self.setup_controller.setup(self.config, use_proxy)

                if success:
                    # Mark environment as used when setup is successfully executed
                    if self.config or self._has_custom_setup(task_config):
                        self.is_environment_used = True
                    break
                else:
                    logger.error(
                        "Environment setup failed, retrying (%d/%d)...",
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    time.sleep(5)
            else:
                break
            
        logger.info("Environment setup complete.")

        observation = self._get_obs()
        return observation

    def _get_obs(self):
        # We provide screenshot, accessibility_tree (optional), terminal (optional), and instruction.
        # can be customized and scaled
        return {
            "screenshot": self.controller.get_screenshot(),
            # "accessibility_tree": self.controller.get_accessibility_tree() if self.require_a11y_tree else None,
            # "terminal": self.controller.get_terminal_output() if self.require_terminal else None,
            "instruction": self.instruction
        }

    def check_health(self):
        return self.controller.check_health()
    
    @property
    def vm_platform(self):
        return self.controller.get_vm_platform()

    @property
    def vm_screen_size(self):
        return self.controller.get_vm_screen_size()

    def _set_task_info(self, task_config: Dict[str, Any]):
        """Set task info (proxy logic is handled in reset method)"""
        self.task_config = task_config
        self.task_id: str = self._task_get(task_config, "id")
        self.cache_dir: str = os.path.join(self.cache_dir_base, self.task_id)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.instruction = self._task_get(task_config, "instruction")
        self.config = self._task_get(task_config, "config", []) or []

        self._set_evaluator_info(task_config)

    def _create_tmp_metric_function(self, code: str) -> Metric:
        """Create a callable metric function from a dynamic code string.

        设计约定：
        - ``code`` 是一段完整的 Python 源码，包含若干 ``import`` 和至少一个函数定义
        - "下面的第一个真正的函数"（第一条顶层 ``def``）就是要用的 metric 函数
        - 一个任务（task）内多个动态 metric 的函数名互不重复，但不同任务之间可以相同

        实现策略：
        - 不写入临时文件，直接在独立 namespace 中 ``exec(code)``
        - 使用 ``ast`` 解析源码，找到第一个顶层 ``FunctionDef`` 的名字
        - 从 namespace 中取出同名可调用对象作为 metric，并缓存到 ``self._dynamic_metric_funcs``
        - 调用 ``_delete_tmp_metric_function`` 时统一清理缓存引用，让 GC 回收
        """
        # 先用 AST 找到第一个函数定义的名字
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid dynamic metric code, failed to parse: {e}")

        func_name: Optional[str] = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break

        if not func_name:
            raise ValueError(f"No function definition found in dynamic metric code: {code}")

        # 在独立 namespace 中执行代码，避免污染当前模块的全局命名空间
        namespace: Dict[str, Any] = {}
        try:
            exec(code, namespace)
        except Exception as e:
            raise RuntimeError(f"Failed to execute dynamic metric code: {e}")

        func = namespace.get(func_name)
        if not callable(func):
            raise ValueError(f"Dynamic metric function '{func_name}' not found or not callable")

        return func

    def _resolve_metric_function(self, func_name: str) -> Metric:
        func = getattr(metrics, func_name, None)
        if callable(func):
            return func

        func = resolve_scalecua_metric(func_name)
        if callable(func):
            return func

        raise AttributeError(f"Metric function '{func_name}' not found")

    def _resolve_getter_function(self, getter_type: str) -> Getter:
        func = getattr(getters, f"get_{getter_type}", None)
        if callable(func):
            return func

        func = resolve_scalecua_getter(getter_type)
        if callable(func):
            return func

        raise AttributeError(f"Getter function 'get_{getter_type}' not found")

    def _set_evaluator_info(self, task_config: Dict[str, Any]):
        """Set evaluator information from task config"""
        # evaluator dict
        # func -> metric function string, or list of metric function strings
        # conj -> conjunction of multiple metrics if func is a list with length > 1, "and"/"or"
        # result -> result getter config, or list of result getter configs
        # expected (optional) -> expected getter config, or list of expected getter configs
        # options (optional) -> metric options, or list of metric options
        # if func is a str list, then result, expected (if exists), options (if exists) should also be lists of the same length
        # even if one of the metrics does not need expected or options field, it should be included in the list with None
        self.metric = None

        evaluator = self._task_get(task_config, "evaluator", None)
        if not evaluator:
            return
        if not isinstance(evaluator, dict):
            logger.warning("Skip evaluator setup because evaluator is not a dict: %s", type(evaluator).__name__)
            return

        self.evaluator = evaluator

        if "func" not in self.evaluator or not self.evaluator["func"]:
            return
        
        dynamic: bool = True if "dynamic" in self.evaluator.keys() else False
        if not dynamic: 
            self.metric: Metric = [self._resolve_metric_function(func) for func in self.evaluator["func"]] \
                if isinstance(self.evaluator["func"], list) \
                else self._resolve_metric_function(self.evaluator["func"])
        else:
            code = [code for code in self.evaluator["code"]] if isinstance(self.evaluator["code"], list) else self.evaluator["code"]
            if code:
                self.metric: Metric|List[Metric] = [self._create_tmp_metric_function(c) for c in code] if isinstance(code, list) else self._create_tmp_metric_function(code)
            
        self.metric_conj: str = self.evaluator.get("conj", "and")  # take conjunction of multiple metrics
        if "result" in self.evaluator and len(self.evaluator["result"]) > 0:
            self.result_getter: Getter = [self._resolve_getter_function(res["type"]) for res in
                                          self.evaluator["result"]] \
                if isinstance(self.evaluator["result"], list) \
                else self._resolve_getter_function(self.evaluator["result"]["type"])
        else:
            self.result_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None

        if "expected" in self.evaluator and len(self.evaluator["expected"]) > 0:
            self.expected_getter: Getter = [self._resolve_getter_function(exp["type"]) if exp else None for exp in
                                            self.evaluator["expected"]] \
                if isinstance(self.evaluator["expected"], list) \
                else self._resolve_getter_function(self.evaluator["expected"]["type"])
        else:
            self.expected_getter = [None] * len(self.metric) \
                if isinstance(self.metric, list) \
                else None
        self.metric_options: Union[List[Dict[str, Any]], Dict[str, Any]] = [opt if opt else {} for opt in
                                                                            self.evaluator["options"]] \
            if isinstance(self.evaluator.get("options", {}), list) \
            else self.evaluator["options"] \
            if "options" in self.evaluator \
            else [{}] * len(self.metric) \
            if isinstance(self.metric, list) \
            else {}

        assert (not isinstance(self.evaluator["func"], list)
                or (len(self.metric) == len(self.result_getter) == len(self.expected_getter) == len(
                    self.metric_options)))

    def step(self, action, pause=2):
        self._step_no += 1
        self.action_history.append(action)
        
        # Mark environment as used when step is called
        self.is_environment_used = True

        reward = 0  # todo: Define reward calculation for each example
        done = False  # todo: Define episode termination condition for each example
        info = {}
        logger.info(f"Step {self._step_no} in trajectory {self._traj_no} with action: {action}")
        # handle the special actions
        if action in ['WAIT', 'FAIL', 'DONE'] or (type(action) == dict and action['action_type'] in ['WAIT', 'FAIL', 'DONE']):
            if action == 'WAIT':
                time.sleep(pause)
            elif action == 'FAIL':
                done = True
                info = {"fail": True}
            elif action == 'DONE':
                done = True
                info = {"done": True}

        if self.action_space == "computer_13":
            # the set of all possible actions defined in the action representation
            self.controller.execute_action(action)
        elif self.action_space == "pyautogui" or self.action_space == "claude_computer_use":
            if action in ['WAIT', 'FAIL', 'DONE']:
                self.controller.execute_action(action)
            elif not action.startswith("pyautogui"):
                pass
            else:
                # the set of all possible python commands insides `pyautogui`
                if type(action) == str:
                    # Fix PyAutoGUI '<' character bug before execution
                    fixed_command = _fix_pyautogui_less_than_bug(action)
                    self.controller.execute_python_command(fixed_command)
                elif type(action) == dict:
                    # Fix PyAutoGUI '<' character bug before execution
                    fixed_command = _fix_pyautogui_less_than_bug(action['command'])
                    self.controller.execute_python_command(fixed_command)

        time.sleep(pause)
        observation = self._get_obs()

        return observation, reward, done, info

    def evaluate(self):
        """
        Evaluate whether the task is successfully completed.
        """
        if getattr(self, "task_config", None) is not None and self._has_custom_evaluate(self.task_config):
            return self._call_task_evaluate(self.task_config)

        if not self.metric:
            return 0
        
        postconfig = self.evaluator.get("postconfig", [])
        self.setup_controller.setup(postconfig, self.enable_proxy)
        # Mark environment as used if there were postconfig setup operations
        if postconfig:
            self.is_environment_used = True

        if self.evaluator['func'] == "infeasible":
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 1
            else:
                return 0
        else:
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 0

        if type(self.metric) == list:
            # Multiple metrics to evaluate whether the task is successfully completed
            results = []
            assert len(self.metric) == len(self.result_getter), "The number of metrics and result getters must be the same"
            if "expected" in self.evaluator:
                assert len(self.metric) == len(self.expected_getter), "The number of metrics and expected getters must be the same"
            for idx, metric in enumerate(self.metric):
                try:
                    config = self.evaluator["result"][idx]
                    result_state = self.result_getter[idx](self, config)
                except FileNotFoundError:
                    logger.error("File not found!")
                    if self.metric_conj == 'and':
                        return 0


                if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                    try:
                        expected_state = self.expected_getter[idx](self, self.evaluator["expected"][idx])
                        metric: int = metric(result_state, expected_state, **self.metric_options[idx])
                    except Exception:
                        metric = 0
                else:
                    try:
                        metric: int = metric(result_state, **self.metric_options[idx])
                    except Exception:
                        metric = 0

                if self.metric_conj == 'and' and float(metric) == 0.0:
                    return 0
                elif self.metric_conj == 'or' and float(metric) == 1.0:
                    return 1
                else:
                    results.append(metric)

            return sum(results) / len(results) if self.metric_conj == 'and' or self.metric_conj == 'avg' else max(results)
        else:
            # Single metric to evaluate whether the task is successfully completed
            try:
                result_state = self.result_getter(self, self.evaluator["result"])
            except FileNotFoundError:
                logger.error("File not found!")
                return 0

            if "expected" in self.evaluator and self.expected_getter and self.evaluator["expected"]:
                try:
                    expected_state = self.expected_getter(self, self.evaluator["expected"])
                    metric: float = self.metric(result_state, expected_state, **self.metric_options)
                except Exception:
                    metric = 0
            else:
                try:
                    metric: float = self.metric(result_state, **self.metric_options)
                except Exception:
                    metric = 0

        return metric

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.controller.get_screenshot()
        else:
            raise ValueError('Unsupported render mode: {}'.format(mode))
