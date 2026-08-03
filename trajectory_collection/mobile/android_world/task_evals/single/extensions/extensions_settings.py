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

"""Settings tasks that configure another app's notification and data usage."""

import re
from typing import Any, Optional

from absl import logging
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval

_GOOGLE_PACKAGE = 'com.google.android.googlequicksearchbox'
_DATA_DOWNLOAD_CHANNEL_ID = 'download-notification-channel-id'

# NotificationManager importance levels. The Settings UI maps "Silent" to
# IMPORTANCE_LOW (2) and "Silent" + "Minimize" to IMPORTANCE_MIN (1).
_IMPORTANCE_MIN = 1


def _get_uid(env: interface.AsyncEnv, package: str) -> Optional[int]:
  """Returns the uid of an installed package."""
  res = adb_utils.issue_generic_request(
      ['shell', 'dumpsys', 'package', package], env.controller
  )
  m = re.search(r'userId=(\d+)', res.generic.output.decode(errors='ignore'))
  return int(m.group(1)) if m else None


def _get_channel_importance(
    env: interface.AsyncEnv, package: str, channel_id: str
) -> Optional[int]:
  """Returns a notification channel's current importance, or None if absent."""
  res = adb_utils.issue_generic_request(
      ['shell', 'dumpsys', 'notification', '--noredact'], env.controller
  )
  out = res.generic.output.decode(errors='ignore')
  in_package_block = False
  for line in out.splitlines():
    stripped = line.strip()
    if stripped.startswith('AppSettings:'):
      in_package_block = stripped.startswith(f'AppSettings: {package} ')
      continue
    if in_package_block and f"mId='{channel_id}'" in stripped:
      m = re.search(r'mImportance=(-?\d+)', stripped)
      if m:
        return int(m.group(1))
  return None


def _background_data_restricted(env: interface.AsyncEnv, uid: int) -> bool:
  """Returns True if metered background data is blocked for the uid."""
  res = adb_utils.issue_generic_request(
      ['shell', 'dumpsys', 'netpolicy'], env.controller
  )
  for line in res.generic.output.decode(errors='ignore').splitlines():
    stripped = line.strip()
    if (
        stripped.startswith(f'UID={uid} policy=')
        and 'REJECT_METERED_BACKGROUND' in stripped
    ):
      return True
  return False


def _reset_background_data_policy(env: interface.AsyncEnv) -> None:
  """Removes the background-data restriction for the Google app, if set.

  `cmd netpolicy remove` exits non-zero when the uid is not blacklisted, so
  only issue it when the restriction is actually present.
  """
  uid = _get_uid(env, _GOOGLE_PACKAGE)
  if uid is not None and _background_data_restricted(env, uid):
    adb_utils.issue_generic_request(
        [
            'shell', 'cmd', 'netpolicy', 'remove',
            'restrict-background-blacklist', str(uid),
        ],
        env.controller,
    )


class SettingsGoogleAppNotificationsAndData(task_eval.TaskEval):
  """Silence and minimize the Google app's Data Download notifications, then
  block its background mobile data.

  Validation reads system state over adb:
  - The 'Data Download Notification Channel' of the Google app must be at
    IMPORTANCE_MIN (the Settings UI's "Silent" + "Minimize").
  - The Google app's uid must carry the REJECT_METERED_BACKGROUND netpolicy
    (the App info "Background data" toggle turned off).

  The netpolicy half is reset in initialize_task / tear_down. The channel
  importance cannot be written from adb, so if it is already at the target
  value when the task starts (e.g. from a previous run on the same emulator
  state), a warning is logged and the reward would over-credit; reload the
  emulator snapshot to fully reset.
  """

  app_names = ('settings',)
  complexity = 4
  schema = {
      'type': 'object',
      'properties': {},
  }
  template = (
      "Configure the Google app so that 'Data Download' notifications are"
      ' silent and minimized, and then prevent the app from using mobile data'
      ' in the background.'
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    _reset_background_data_policy(env)
    importance = _get_channel_importance(
        env, _GOOGLE_PACKAGE, _DATA_DOWNLOAD_CHANNEL_ID
    )
    if importance is not None and importance <= _IMPORTANCE_MIN:
      logging.warning(
          'Data Download channel already at importance %d before the task'
          ' started; the notification check is pre-satisfied. Reload the'
          ' emulator snapshot for a clean run.',
          importance,
      )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    importance = _get_channel_importance(
        env, _GOOGLE_PACKAGE, _DATA_DOWNLOAD_CHANNEL_ID
    )
    if importance is None or importance > _IMPORTANCE_MIN:
      return 0.0
    uid = _get_uid(env, _GOOGLE_PACKAGE)
    if uid is None or not _background_data_restricted(env, uid):
      return 0.0
    return 1.0

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    _reset_background_data_policy(env)
