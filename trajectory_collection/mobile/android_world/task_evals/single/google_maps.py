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

"""Tasks for the Google Maps app."""

import random
from typing import Any

from absl import logging
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval

_APP_NAME = 'google maps'
_PACKAGE_NAME = 'com.google.android.apps.maps'

_LOCATIONS = [
    'Eiffel Tower, Paris',
    'Statue of Liberty, New York',
    'Big Ben, London',
    'Sydney Opera House, Sydney',
    'Colosseum, Rome',
    'Golden Gate Bridge, San Francisco',
    'Taj Mahal, Agra',
    'Central Park, New York',
    'Times Square, New York',
    'Tokyo Tower, Tokyo',
    'Buckingham Palace, London',
    'Empire State Building, New York',
    'Hollywood Sign, Los Angeles',
    'Grand Canyon, Arizona',
    'Niagara Falls',
]


class _GoogleMaps(task_eval.TaskEval):
  """Base class for Google Maps tasks."""

  app_names = (_APP_NAME,)

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.launch_app(_APP_NAME, env.controller)

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    adb_utils.close_app(_APP_NAME, env.controller)

  def _is_maps_foreground(self, env: interface.AsyncEnv) -> bool:
    """Check if Google Maps is the foreground app."""
    active_activity, _ = adb_utils.get_current_activity(env.controller)
    return _PACKAGE_NAME in active_activity


class GoogleMapSearchLocation(_GoogleMaps):
  """Task for searching a location on Google Maps."""

  complexity = 1
  schema = {
      'type': 'object',
      'properties': {
          'location': {'type': 'string'},
      },
      'required': ['location'],
  }
  template = 'Open Google Maps and search for {location}.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._is_maps_foreground(env):
      return 1.0
    logging.info('Google Maps is not the foreground app.')
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'location': random.choice(_LOCATIONS)}


class GoogleMapNavigateToLocation(_GoogleMaps):
  """Task for navigating to a location on Google Maps."""

  complexity = 2
  schema = {
      'type': 'object',
      'properties': {
          'location': {'type': 'string'},
      },
      'required': ['location'],
  }
  template = 'Open Google Maps and start navigation to {location}.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if self._is_maps_foreground(env):
      return 1.0
    logging.info('Google Maps is not the foreground app.')
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'location': random.choice(_LOCATIONS)}


# --- 以下为新增任务 ---

class GoogleMapFindHighRatedLateNightRestaurant(_GoogleMaps):
  """寻找深夜高分餐厅。"""

  complexity = 3
  schema = {'type': 'object', 'properties': {}}
  template = 'Find the restaurant within 1 km, who has the highest rate (at least 4.0) and open on 2:00 a.m.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapParksPublicTransit(_GoogleMaps):
  """公园间的公共交通。"""

  complexity = 3
  schema = {'type': 'object', 'properties': {}}
  template = 'How can I transport between the nearest two parks around me? Provide the best public transportation choice.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapSaveTopBars(_GoogleMaps):
  """保存最好的三个酒吧。"""

  complexity = 3
  schema = {'type': 'object', 'properties': {}}
  template = 'Add the best three bars within 1 km into my save.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapShortestHistoryDistance(_GoogleMaps):
  """查找搜索历史中距离最短的两个位置。"""

  complexity = 2
  schema = {'type': 'object', 'properties': {}}
  template = 'Which two locations have the shortest distance in my search history?'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapHospitalAndGasRoute(_GoogleMaps):
  """前往医院并在途中加油。"""

  complexity = 4
  schema = {'type': 'object', 'properties': {}}
  template = 'I want to go to the nearest hospital and fill my gas tank along the road. Provide me a best route for driving my own car.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapCheapestHighRatedHotel(_GoogleMaps):
  """寻找最便宜的高分酒店。"""

  complexity = 3
  schema = {'type': 'object', 'properties': {}}
  template = 'Provide me the cheapest hotel within 1 km, whose rate is above 4.0.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapHotelPriceRange(_GoogleMaps):
  """查询下周酒店价格范围。"""

  complexity = 3
  schema = {'type': 'object', 'properties': {}}
  template = "If I want to book a hotel nearby from next Monday to next Friday, what's the price range?"

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapAccessibleAttractions(_GoogleMaps):
  """寻找适合腿部受伤人士的景点。"""

  complexity = 2
  schema = {'type': 'object', 'properties': {}}
  template = 'I have broken my legs and want to go to some attractions nearby, please give me some destination.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapDownloadOffline(_GoogleMaps):
  """下载离线地图。"""

  complexity = 2
  schema = {'type': 'object', 'properties': {}}
  template = 'Download the offline map of current location.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapFoodSearchHistory(_GoogleMaps):
  """查看历史搜索中的饮食记录。"""

  complexity = 2
  schema = {'type': 'object', 'properties': {}}
  template = 'What Food&Drink have I looked for in the history?'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapLauPaSatDinnerPlan(_GoogleMaps):
  complexity = 3
  schema = {'type': 'object', 'properties': {}}
  template = "Search for 'Satay' at 'Lau Pa Sat' in Singapore on Google Maps. Find a stall that has a rating of 4.5 stars or higher. Copy the stall name and look at the 'Popular Times' graph to see when it gets busy on Wednesday (today). Then, open Markor, create a new file named Dinner_Plan.md, and record the stall name and a brief note about its peak hours."

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapGardensTripCalendar(_GoogleMaps):
  complexity = 3
  schema = {'type': 'object', 'properties': {}}
  template = "Find 'Gardens by the Bay' on Google Maps and check the public transport route from 'Boon Lay MRT Station'. Note the total travel time for the fastest route. Then, open Simple Calendar Pro and create an event for tomorrow at 10:00 AM titled 'Gardens Trip', and in the event description, write: 'Total travel time is [the time you found] via MRT/Bus'."

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapSingaporeDistanceCalc(_GoogleMaps):
  complexity = 3.5
  schema = {'type': 'object', 'properties': {}}
  template = "Look up the distances from your current location in Singapore to 'Changi Airport Terminal 4' and 'Woodlands Checkpoint' using Google Maps. Open the Calculator app to find the difference in distance (kilometers) between these two locations. Finally, record both distances and the calculated difference in a new Markor file named Distance_Calc.txt."

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapOrchardClinicAlarm(_GoogleMaps):
  complexity = 3
  schema = {'type': 'object', 'properties': {}}
  template = "Search for a '24-hour Clinic' near 'Orchard Road' on Google Maps. Pick one that is highly rated, copy its full address and phone number. Then, open the Clock app and set an alarm for 30 minutes from now with the label 'Call Clinic' to remind me to check their availability."

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class GoogleMapVivoCityShoppingEmail(_GoogleMaps):
  complexity = 3
  schema = {'type': 'object', 'properties': {}}
  template = "Find the 'VivoCity' shopping mall on Google Maps and identify its earliest opening time for tomorrow. Also, find a highly-rated 'Cafe' inside or very close to the mall. Open Gmail and draft an email to your friend Alex with the subject 'Shopping Trip' stating the mall's opening time and the name of the cafe you found for breakfast."

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return 1.0 if self._is_maps_foreground(env) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}
