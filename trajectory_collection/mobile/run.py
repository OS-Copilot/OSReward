# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run eval suite.

The run.py module is used to run a suite of tasks, with configurable task
combinations, environment setups, and agent configurations. You can run specific
tasks or all tasks in the suite and customize various settings using the
command-line flags.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import model_profiles
from android_world.agents import seeact
from android_world.env import env_launcher
from android_world.env import interface

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


def _load_dotenv_if_present(dotenv_path: str = '.env') -> None:
  """Loads KEY=VALUE pairs from a local .env file into environment variables."""
  if not os.path.isfile(dotenv_path):
    return
  with open(dotenv_path, 'r', encoding='utf-8') as f:
    for raw_line in f:
      line = raw_line.strip()
      if not line or line.startswith('#') or '=' not in line:
        continue
      key, value = line.split('=', 1)
      key = key.strip()
      value = value.strip()
      if (
          len(value) >= 2
          and value[0] == value[-1]
          and value[0] in {'"', "'"}
      ):
        value = value[1:-1]
      # Keep real process env as highest priority.
      os.environ.setdefault(key, value)


_load_dotenv_if_present()


def _find_adb_directory() -> str:
  """Returns the directory where adb is located."""
  potential_paths = [
      os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
      os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
  ]
  # Windows: ANDROID_HOME or default SDK location
  android_home = os.environ.get(
      'ANDROID_HOME') or os.environ.get('ANDROID_SDK_ROOT')
  if android_home:
      potential_paths.append(os.path.join(
          android_home, 'platform-tools', 'adb.exe'))
      potential_paths.append(os.path.join(
          android_home, 'platform-tools', 'adb'))
  localappdata = os.environ.get('LOCALAPPDATA')
  if localappdata:
      potential_paths.append(os.path.join(
          localappdata, 'Android', 'Sdk', 'platform-tools', 'adb.exe'))
  for path in potential_paths:
      if path and os.path.isfile(path):
          return path
  return ''


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)
_FREEZE_DATETIME = flags.DEFINE_boolean(
    'freeze_datetime',
    False,
    'Whether to freeze the emulator datetime for reproducibility. Set False to'
    ' keep real device time during task execution.',
)

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_EXT_FAMILY,
    [
        registry.TaskRegistry.ANDROID_WORLD_EXT_FAMILY,
    ],
    'Suite family to run. See registry.py for more information.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 30, 'Random seed for task randomness.'
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from. If the'
    ' directory contains existing checkpoint files, evaluation will resume from'
    ' the latest checkpoint. If the directory is empty or does not exist, a new'
    ' directory will be created.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    'runs',
    'The path to save results to if not resuming from a checkpoint is not'
    ' provided.',
)

# Agent specific.
_AGENT_NAME = flags.DEFINE_string(
    'agent_name', os.environ.get('AGENT_NAME', 'toolcall'), help='Agent name.'
)

# Tool-call agent, against any OpenAI-compatible server.
_MODEL_BASE_URL = flags.DEFINE_string(
    'model_base_url',
    os.environ.get(
        'MODEL_BASE_URL',
        os.environ.get('QWEN3VL_MODEL_BASE_URL', 'http://127.0.0.1:8000/v1'),
    ),
    'OpenAI-compatible base_url, e.g. http://host:port/v1',
)
_MODEL_API_KEY = flags.DEFINE_string(
    'model_api_key',
    os.environ.get(
        'MODEL_API_KEY', os.environ.get('QWEN3VL_MODEL_API_KEY', 'EMPTY')
    ),
    'API key for the OpenAI-compatible server (if needed).',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    os.environ.get('MODEL_NAME', os.environ.get('QWEN3VL_MODEL_NAME', '')),
    'Model id passed to /v1/chat/completions. Must be listed in'
    ' android_world/agents/model_profiles.py, or paired with --model_profile.',
)
_MODEL_PROFILE = flags.DEFINE_string(
    'model_profile',
    os.environ.get('MODEL_PROFILE', ''),
    'Force a prompt format for a model not listed in model_profiles.py.'
    " One of: 'qwen3vl', 'gemini3'. Normally left unset.",
)

# Deprecated aliases, kept so existing .env files and launch configs keep
# working. Prefer the --model_* flags above.
flags.DEFINE_alias('qwen3vl_model_base_url', 'model_base_url')
flags.DEFINE_alias('qwen3vl_model_api_key', 'model_api_key')
flags.DEFINE_alias('qwen3vl_model_name', 'model_name')

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed when running multiple task combinations'
    ' (n_task_combinations > 1).',
)



def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Gets agent."""
  print('Initializing agent...')
  # Model-agnostic tool-call agent. 'qwen3vl' is the historical name.
  if _AGENT_NAME.value in ('toolcall', 'qwen3vl'):
    agent = seeact.ToolCallAgent(
        env,
        model_base_url=_MODEL_BASE_URL.value,
        model_api_key=_MODEL_API_KEY.value,
        model_name=_MODEL_NAME.value,
        model_profile=_MODEL_PROFILE.value,
    )
  else:
    raise ValueError(f'Unknown agent: {_AGENT_NAME.value}')

  # Label results by the model actually evaluated, not by the agent wrapper.
  if _MODEL_NAME.value:
    agent.name = _MODEL_NAME.value
  else:
    agent.name = _AGENT_NAME.value

  return agent


def _main() -> None:
  """Runs eval suite and gets rewards back."""
  # Resolve the model profile up front so an unknown model fails here, rather
  # than after the emulator has been booted and the suite instantiated.
  if _AGENT_NAME.value in ('toolcall', 'qwen3vl'):
    model_profiles.resolve(_MODEL_NAME.value, _MODEL_PROFILE.value)

  # Share datetime policy with task initialization logic.
  os.environ['ANDROID_WORLD_FREEZE_DATETIME'] = (
      '1' if _FREEZE_DATETIME.value else '0'
  )
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      freeze_datetime=_FREEZE_DATETIME.value,
      adb_path=_ADB_PATH.value,
  )

  n_task_combinations = _N_TASK_COMBINATIONS.value
  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family=_SUITE_FAMILY.value),
      n_task_combinations=n_task_combinations,
      seed=_TASK_RANDOM_SEED.value,
      tasks=_TASKS.value,
      use_identical_params=_FIXED_TASK_SEED.value,
  )
  suite.suite_family = _SUITE_FAMILY.value

  agent = _get_agent(env, _SUITE_FAMILY.value)

  agent.transition_pause = None

  if _CHECKPOINT_DIR.value:
    checkpoint_dir = _CHECKPOINT_DIR.value
  else:
    checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)

  print(
      f'Starting eval with agent {_AGENT_NAME.value} and writing to'
      f' {checkpoint_dir}'
  )
  suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
  )
  print(
      f'Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value}'
      f' family. Wrote to {checkpoint_dir}.'
  )
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
