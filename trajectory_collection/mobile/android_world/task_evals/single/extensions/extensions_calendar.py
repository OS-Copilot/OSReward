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

"""Additional tasks for Simple Calendar Pro app."""

import dataclasses
import random
from typing import Any

from android_world.env import device_constants
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single.calendar import calendar_utils
from android_world.task_evals.single.calendar import events_generator
from android_world.task_evals.single.calendar.calendar import (
    SimpleCalendarAddOneEvent,
    SimpleCalendarDeleteEvents,
    SimpleCalendarDeleteEventsOnRelativeDay,
    generate_noise_events,
    _SimpleCalendar,
    _YEAR,
    _MONTH,
    _DAY,
    _DAY_OF_WEEK,
    _HOUR,
    EVENT_TITLE,
    _EVENT_DESCRIPTION,
    _DURATION_MINS,
)
from android_world.utils import datetime_utils


class SimpleCalendarAddOneEventNextWeek(SimpleCalendarAddOneEvent):
  """Task for creating a calendar event in Simple Calendar Pro exactly one week from today."""

  complexity = 3.4
  template = (
      "In Simple Calendar Pro, create a calendar event for next week on"
      " {day_of_week}, {year}-{month}-{day}, at {hour}h with the title"
      " '{event_title}' and the description '{event_description}'. The event"
      " should last for {duration_mins} mins."
  )

  @classmethod
  def _get_random_target_row(cls):
    # Generate an event for exactly 7 days from today.
    target_day_offset = 7
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            device_constants.DT.day + target_day_offset,
            device_constants.DT.day + target_day_offset,
        )
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    # Add day of week for the template
    dt = params[sqlite_validators.ROW_OBJECTS][0].start_datetime
    params[_DAY_OF_WEEK] = dt.strftime("%A")
    return params


class SimpleCalendarAddOneEventInOneMonth(SimpleCalendarAddOneEvent):
  """Task for creating a calendar event in Simple Calendar Pro roughly one month from today."""

  complexity = 3.4
  template = (
      "In Simple Calendar Pro, create a calendar event in one month from today"
      " ({year}-{month}-{day}) at {hour}h with the title '{event_title}' and"
      " the description '{event_description}'. The event should last for"
      " {duration_mins} mins."
  )

  @classmethod
  def _get_random_target_row(cls):
    # Generate an event for 30 days from today.
    target_day_offset = 30
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            device_constants.DT.day + target_day_offset,
            device_constants.DT.day + target_day_offset,
        )
    )


class SimpleCalendarAddOneEventThisWeekend(SimpleCalendarAddOneEvent):
  """Task for creating a calendar event for this coming weekend (Saturday or Sunday)."""

  complexity = 3.4
  template = (
      "In Simple Calendar Pro, create a calendar event for this weekend on"
      " {day_of_week} at {hour}h with the title '{event_title}' and the"
      " description '{event_description}'. The event should last for"
      " {duration_mins} mins."
  )

  @classmethod
  def _get_random_target_row(cls):
    # Calculate days until next Saturday (6) or Sunday (7)
    # isoweekday: Mon=1 ... Sun=7
    current_weekday = device_constants.DT.isoweekday()
    days_until_saturday = 6 - current_weekday
    if days_until_saturday < 0:  # If today is Sunday, look at next Saturday?
       # Assuming "this weekend" implies the upcoming one. 
       # If today is Sunday, "this weekend" might refer to today, but let's aim for next Sat/Sun to be safe or today if it is Sun.
       # For simplicity, let's just ensure we pick a Sat/Sun within the next 7 days.
       days_until_saturday += 7
    
    # Randomly choose Saturday or Sunday
    offset = days_until_saturday + random.choice([0, 1])
    
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            device_constants.DT.day + offset,
            device_constants.DT.day + offset,
        )
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    dt = params[sqlite_validators.ROW_OBJECTS][0].start_datetime
    params[_DAY_OF_WEEK] = dt.strftime("%A")
    return params


class SimpleCalendarAddOneEventToday(SimpleCalendarAddOneEvent):
  """Task for creating a calendar event for the current day."""

  complexity = 3.0
  template = (
      "In Simple Calendar Pro, create a calendar event for today at {hour}h"
      " with the title '{event_title}' and the description"
      " '{event_description}'. The event should last for {duration_mins} mins."
  )

  @classmethod
  def _get_random_target_row(cls):
    return events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            device_constants.DT.day, device_constants.DT.day
        )
    )


class SimpleCalendarAddMorningEvent(SimpleCalendarAddOneEvent):
  """Task for creating a calendar event specifically in the morning (6 AM - 11 AM)."""

  complexity = 3.4
  template = (
      "In Simple Calendar Pro, create a morning meeting on {year}-{month}-{day}"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. The event should last for {duration_mins} mins."
  )

  @classmethod
  def _get_random_target_row(cls):
    # Force hour to be between 6 and 11
    hour = random.randint(6, 11)
    start_ts = datetime_utils._create_unix_ts(
        day=random.randint(1, 30),
        hour=hour,
        month=device_constants.DT.month,
        year=device_constants.DT.year,
        timezone="UTC",
    )
    return events_generator.generate_event(start_ts)


class SimpleCalendarAddEveningEvent(SimpleCalendarAddOneEvent):
  """Task for creating a calendar event specifically in the evening (6 PM - 9 PM)."""

  complexity = 3.4
  template = (
      "In Simple Calendar Pro, create an evening event on {year}-{month}-{day}"
      " at {hour}h with the title '{event_title}' and the description"
      " '{event_description}'. The event should last for {duration_mins} mins."
  )

  @classmethod
  def _get_random_target_row(cls):
    # Force hour to be between 18 (6 PM) and 21 (9 PM)
    hour = random.randint(18, 21)
    start_ts = datetime_utils._create_unix_ts(
        day=random.randint(1, 30),
        hour=hour,
        month=device_constants.DT.month,
        year=device_constants.DT.year,
        timezone="UTC",
    )
    return events_generator.generate_event(start_ts)


class SimpleCalendarAddLongEvent(SimpleCalendarAddOneEvent):
  """Task for creating a calendar event with a long duration (2 hours)."""

  complexity = 3.4
  template = (
      "In Simple Calendar Pro, create a long calendar event on"
      " {year}-{month}-{day} at {hour}h with the title '{event_title}' and the"
      " description '{event_description}'. The event should last for 2 hours."
  )

  @classmethod
  def _get_random_target_row(cls):
    event = events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts()
    )
    # Force duration to 120 minutes
    new_end_ts = event.start_ts + (120 * 60)
    return dataclasses.replace(event, end_ts=new_end_ts)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    # Ensure duration_mins param reflects the forced 120 mins
    params[_DURATION_MINS] = 120
    return params


class SimpleCalendarDeleteEventsTomorrow(SimpleCalendarDeleteEvents):
  """Task to delete all calendar events scheduled for tomorrow."""

  complexity = 1.4
  template = (
      "In Simple Calendar Pro, delete all the calendar events scheduled for"
      " tomorrow ({year}-{month}-{day})."
  )

  @classmethod
  def _get_random_target_row(cls, day: int):
    # This method is called by generate_random_params with a day,
    # but we want to enforce 'tomorrow' relative to device date.
    # However, generate_random_params in the base class controls the loop.
    # We override generate_random_params to ensure we target tomorrow.
    pass 

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    target_day = device_constants.DT.day + 1
    
    # Generate target events for tomorrow
    events = [
        events_generator.generate_event(
            datetime_utils.create_random_october_2023_unix_ts(
                start_day=target_day, end_day=target_day
            )
        )
        for _ in range(cls.n_rows)
    ]
    
    # Generate noise events (not on target day)
    noise_events = generate_noise_events(
        events,
        cls.n_rows_noise,
        filter_fn=lambda candidate: candidate.start_datetime.day != target_day,
    )
    
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: target_day,
        sqlite_validators.ROW_OBJECTS: events,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
    }


class SimpleCalendarDeleteEventsNextWeek(SimpleCalendarDeleteEventsOnRelativeDay):
  """Task to delete all events on a specific day next week."""

  complexity = 1.4
  template = (
      "In Simple Calendar Pro, delete all events scheduled for next"
      " {day_of_week}."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Target a day 7-13 days from now
    start_offset = 7
    end_offset = 13
    
    template = events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts(
            start_day=device_constants.DT.day + start_offset,
            end_day=device_constants.DT.day + end_offset,
        )
    )
    
    target_day = template.start_datetime.day
    
    events = [
        cls._get_random_target_row(target_day)
        for _ in range(cls.n_rows)
    ]
    
    noise_events = generate_noise_events(
        events,
        cls.n_rows_noise,
        filter_fn=lambda candidate: candidate.start_datetime.day != target_day,
    )
    
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: target_day,
        _DAY_OF_WEEK: template.start_datetime.strftime("%A"),
        sqlite_validators.ROW_OBJECTS: events,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
    }


class SimpleCalendarDeleteEventsThisWeekend(SimpleCalendarDeleteEventsOnRelativeDay):
  """Task to delete all events on this coming weekend (Saturday or Sunday)."""

  complexity = 1.4
  template = (
      "In Simple Calendar Pro, delete all events scheduled for this upcoming"
      " {day_of_week}."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Calculate days until next Saturday
    current_weekday = device_constants.DT.isoweekday()
    days_until_saturday = 6 - current_weekday
    if days_until_saturday < 0:
       days_until_saturday += 7
       
    offset = days_until_saturday + random.choice([0, 1])
    target_day = device_constants.DT.day + offset
    
    # Generate template to get the weekday name
    template_ts = datetime_utils._create_unix_ts(
        day=target_day, hour=12, month=device_constants.DT.month, year=device_constants.DT.year, timezone="UTC"
    )
    # Re-use helper to generate multiple events on that day
    events = [
        events_generator.generate_event(
            datetime_utils.create_random_october_2023_unix_ts(
                start_day=target_day, end_day=target_day
            )
        )
        for _ in range(cls.n_rows)
    ]
    
    noise_events = generate_noise_events(
        events,
        cls.n_rows_noise,
        filter_fn=lambda candidate: candidate.start_datetime.day != target_day,
    )
    
    return {
        _YEAR: device_constants.DT.year,
        _MONTH: device_constants.DT.month,
        _DAY: target_day,
        _DAY_OF_WEEK: events[0].start_datetime.strftime("%A"),
        sqlite_validators.ROW_OBJECTS: events,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
    }