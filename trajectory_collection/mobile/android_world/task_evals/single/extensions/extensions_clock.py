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

"""Additional tasks for the clock app."""

import random

from android_world.task_evals.single.clock import ClockTimerEntry


class ClockTimerEntryFiveMinutes(ClockTimerEntry):
  """Task for creating a 5-minute timer."""

  complexity = 1
  template = (
      "Create a timer for 5 minutes. Do not start the timer."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 0,
        "minutes": 5,
        "seconds": 0,
    }


class ClockTimerEntryFifteenMinutes(ClockTimerEntry):
  """Task for creating a 15-minute timer."""

  complexity = 1
  template = (
      "Create a timer for 15 minutes. Do not start the timer."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 0,
        "minutes": 15,
        "seconds": 0,
    }


class ClockTimerEntryThirtyMinutes(ClockTimerEntry):
  """Task for creating a 30-minute timer."""

  complexity = 1
  template = (
      "Create a timer for 30 minutes. Do not start the timer."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 0,
        "minutes": 30,
        "seconds": 0,
    }


class ClockTimerEntryOneHour(ClockTimerEntry):
  """Task for creating a 1-hour timer."""

  complexity = 1
  template = (
      "Create a timer for 1 hour. Do not start the timer."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 1,
        "minutes": 0,
        "seconds": 0,
    }


class ClockTimerEntryOneHourThirtyMinutes(ClockTimerEntry):
  """Task for creating a 1.5-hour timer."""

  complexity = 1.2
  template = (
      "Create a timer for 1 hour and 30 minutes. Do not start the timer."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 1,
        "minutes": 30,
        "seconds": 0,
    }


class ClockTimerEntryFortyFiveSeconds(ClockTimerEntry):
  """Task for creating a short 45-second timer."""

  complexity = 1.2
  template = (
      "Create a timer for 45 seconds. Do not start the timer."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 0,
        "minutes": 0,
        "seconds": 45,
    }


class ClockTimerEntryPreciseCooking(ClockTimerEntry):
  """Task for creating a timer with specific minutes and seconds (e.g., for cooking)."""

  complexity = 1.4
  template = (
      "Create a timer for 8 minutes and 30 seconds. Do not start the timer."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 0,
        "minutes": 8,
        "seconds": 30,
    }


class ClockTimerEntryLongDuration(ClockTimerEntry):
  """Task for creating a timer with a long duration (randomized between 5-10 hours)."""

  complexity = 1.4
  template = (
      "Create a timer for {hours} hours. Do not start the timer."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    hours = random.randint(5, 10)
    return {
        "hours": hours,
        "minutes": 0,
        "seconds": 0,
    }


class ClockTimerEntryComplex(ClockTimerEntry):
  """Task for creating a timer with non-zero hours, minutes, and seconds."""

  complexity = 2
  template = (
      "Create a timer for {hours} hours, {minutes} minutes, and {seconds}"
      " seconds. Do not start the timer."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": random.randint(1, 5),
        "minutes": random.randint(1, 59),
        "seconds": random.randint(1, 59),
    }


class ClockTimerEntryMaxTime(ClockTimerEntry):
  """Task for creating a timer for the maximum time in a day (23h 59m 59s)."""

  complexity = 2
  template = (
      "Create a timer for 23 hours, 59 minutes, and 59 seconds. Do not start"
      " the timer."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, int]:
    return {
        "hours": 23,
        "minutes": 59,
        "seconds": 59,
    }