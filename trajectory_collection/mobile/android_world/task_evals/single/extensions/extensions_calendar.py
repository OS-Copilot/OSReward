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
import copy
import datetime
from android_world.task_evals.single.calendar import calendar
from android_world.task_evals.utils import sqlite_schema_utils


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


# =============================================================================
# Phase 2 tasks (previously extensions_calendar2.py).
# =============================================================================

# Keys in generated parameters and used to populate goal template.
# Extended keys
_LOCATION = "location"
_DURATION_HOURS = "duration_hours"
_END_HOUR = "end_hour"


def _add_location_to_event(
    event: sqlite_schema_utils.CalendarEvent,
) -> sqlite_schema_utils.CalendarEvent:
  """Helper to add a location to a generated event."""
  locations = [
      "Conference Room A",
      "Main Hall",
      "Coffee Shop",
      "Home Office",
      "Client Site",
      "Building 42",
      "Downtown",
  ]
  # We assume the dataclass supports 'location'.
  # Based on calendar_evaluators.py, 'location' is a valid field.
  return dataclasses.replace(event, location=random.choice(locations))


class SimpleCalendarAddEventWithLocation(calendar.SimpleCalendarAddOneEvent):
  """Task for creating a calendar event with a specific location."""

  complexity = 4.0
  max_steps = 15
  template = (
      "In Simple Calendar Pro, create a calendar event on {year}-{month}-{day}"
      " at {hour}h with the title '{event_title}', the description"
      " '{event_description}', and the location '{location}'. The event should"
      " last for {duration_mins} mins."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    # Modify the target row to include location
    target_event = params[sqlite_validators.ROW_OBJECTS][0]
    target_event = _add_location_to_event(target_event)
    params[sqlite_validators.ROW_OBJECTS] = [target_event]
    params[_LOCATION] = target_event.location
    return params


class SimpleCalendarAddTwoEventsSameDay(calendar.SimpleCalendarAddOneEvent):
  """Task for creating two different events on the same day."""

  n_rows = 2
  complexity = 5.0
  max_steps = 20
  template = (
      "In Simple Calendar Pro, create two events on {year}-{month}-{day}. "
      "First, create '{event_title_1}' at {hour_1}h lasting {duration_mins_1} "
      "mins. Then, create '{event_title_2}' at {hour_2}h lasting "
      "{duration_mins_2} mins."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Generate timestamp for a specific day
    base_ts = datetime_utils.create_random_october_2023_unix_ts()
    day_ts = datetime.datetime.fromtimestamp(base_ts)
    
    # Event 1 (Morning)
    ts1 = day_ts.replace(hour=9, minute=0).timestamp()
    event1 = events_generator.generate_event(int(ts1))
    
    # Event 2 (Afternoon)
    ts2 = day_ts.replace(hour=14, minute=0).timestamp()
    event2 = events_generator.generate_event(int(ts2))

    # Ensure titles are distinct
    while event2.title == event1.title:
      event2 = events_generator.generate_event(int(ts2))

    return {
        "year": day_ts.year,
        "month": day_ts.month,
        "day": day_ts.day,
        "event_title_1": event1.title,
        "hour_1": event1.start_datetime.hour,
        "duration_mins_1": event1.duration_mins,
        "event_title_2": event2.title,
        "hour_2": event2.start_datetime.hour,
        "duration_mins_2": event2.duration_mins,
        sqlite_validators.ROW_OBJECTS: [event1, event2],
        sqlite_validators.NOISE_ROW_OBJECTS: calendar.generate_noise_events(
            [event1, event2], random.randint(0, 10)
        ),
    }


class SimpleCalendarAddEventWithStartEnd(calendar.SimpleCalendarAddOneEvent):
  """Task for creating an event by specifying start and end hours."""

  complexity = 4.2
  max_steps = 15
  template = (
      "In Simple Calendar Pro, schedule '{event_title}' on {year}-{month}-{day}"
      " from {hour}h to {end_hour}h."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    event = params[sqlite_validators.ROW_OBJECTS][0]
    
    # Ensure duration is at least 1 hour for clean "Start to End" integer hours
    start_dt = event.start_datetime
    new_duration = random.randint(1, 4) * 60  # 1 to 4 hours
    end_dt = start_dt + datetime.timedelta(minutes=new_duration)
    
    # Update event
    event = dataclasses.replace(
        event, end_ts=int(end_dt.timestamp())
    )
    
    params[sqlite_validators.ROW_OBJECTS] = [event]
    params[_DURATION_MINS] = event.duration_mins
    params[_END_HOUR] = end_dt.hour
    return params


class SimpleCalendarDeleteEventByTitle(calendar.SimpleCalendarDeleteOneEvent):
  """Delete an event based on its title, ignoring date/time in instruction."""

  complexity = 4.0
  max_steps = 15
  template = (
      "In Simple Calendar Pro, find and delete the event titled '{event_title}'."
  )
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Generate random params but ensure title is unique in the noise
    params = super().generate_random_params()
    target_event = params[sqlite_validators.ROW_OBJECTS][0]
    
    # Ensure noise events don't have the same title
    noise = params[sqlite_validators.NOISE_ROW_OBJECTS]
    filtered_noise = [n for n in noise if n.title != target_event.title]
    params[sqlite_validators.NOISE_ROW_OBJECTS] = filtered_noise
    
    return params


class SimpleCalendarDeleteAllEventsInWeek(calendar.SimpleCalendarDeleteEvents):
  """Delete all events in a specific week."""

  complexity = 4.5
  max_steps = 20
  n_rows = 5
  template = (
      "In Simple Calendar Pro, delete all events scheduled between "
      "{start_year}-{start_month}-{start_day} and {end_year}-{end_month}-{end_day}."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Target week: Oct 16 (Mon) to Oct 22 (Sun)
    start_date = datetime.datetime(2023, 10, 16)
    end_date = datetime.datetime(2023, 10, 22)
    
    events_in_week = []
    for _ in range(cls.n_rows):
        rand_day = random.randint(16, 22)
        events_in_week.append(calendar.SimpleCalendarDeleteEvents._get_random_target_row(rand_day))

    # Noise outside this week
    noise_events = calendar.generate_noise_events(
        events_in_week,
        20,
        filter_fn=lambda x: not (16 <= x.start_datetime.day <= 22)
    )

    return {
        "start_year": start_date.year,
        "start_month": start_date.month,
        "start_day": start_date.day,
        "end_year": end_date.year,
        "end_month": end_date.month,
        "end_day": end_date.day,
        sqlite_validators.ROW_OBJECTS: events_in_week,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
    }


class SimpleCalendarAddThreeEventsDifferentDays(calendar.SimpleCalendarAddOneEvent):
  """Add three events on three consecutive days."""
  
  n_rows = 3
  complexity = 5.5
  max_steps = 25
  template = (
      "In Simple Calendar Pro, create three events. "
      "1. '{title1}' on {y1}-{m1}-{d1} at {h1}h ({dur1}m). "
      "2. '{title2}' on {y2}-{m2}-{d2} at {h2}h ({dur2}m). "
      "3. '{title3}' on {y3}-{m3}-{d3} at {h3}h ({dur3}m)."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    start_day = random.randint(1, 25)
    events = []
    for i in range(3):
        ts = datetime_utils.create_random_october_2023_unix_ts(
            start_day=start_day + i, end_day=start_day + i
        )
        events.append(events_generator.generate_event(ts))

    return {
        "title1": events[0].title, "y1": 2023, "m1": 10, "d1": events[0].start_datetime.day, "h1": events[0].start_datetime.hour, "dur1": events[0].duration_mins,
        "title2": events[1].title, "y2": 2023, "m2": 10, "d2": events[1].start_datetime.day, "h2": events[1].start_datetime.hour, "dur2": events[1].duration_mins,
        "title3": events[2].title, "y3": 2023, "m3": 10, "d3": events[2].start_datetime.day, "h3": events[2].start_datetime.hour, "dur3": events[2].duration_mins,
        sqlite_validators.ROW_OBJECTS: events,
        sqlite_validators.NOISE_ROW_OBJECTS: calendar.generate_noise_events(events, 10),
    }


class SimpleCalendarDeleteEventByLocation(calendar.SimpleCalendarDeleteOneEvent):
  """Delete an event specifying its location."""

  complexity = 4.2
  max_steps = 15
  template = (
      "In Simple Calendar Pro, delete the event happening at '{location}' on "
      "{year}-{month}-{day}."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    target = params[sqlite_validators.ROW_OBJECTS][0]
    target = _add_location_to_event(target)
    params[sqlite_validators.ROW_OBJECTS] = [target]
    params[_LOCATION] = target.location
    return params


class SimpleCalendarAddAllDayEvent(calendar.SimpleCalendarAddOneEvent):
  """Add an event that effectively spans the work day (approximating all-day)."""
  
  complexity = 4.0
  max_steps = 15
  template = (
      "In Simple Calendar Pro, create an event '{event_title}' on {year}-{month}-{day} "
      "that lasts the entire work day (9am to 5pm)."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    event = params[sqlite_validators.ROW_OBJECTS][0]
    
    # Set to 9am - 5pm
    start_dt = event.start_datetime.replace(hour=9, minute=0)
    end_dt = event.start_datetime.replace(hour=17, minute=0)
    
    event = dataclasses.replace(
        event,
        start_ts=int(start_dt.timestamp()),
        end_ts=int(end_dt.timestamp())
    )
    
    params[sqlite_validators.ROW_OBJECTS] = [event]
    params[_DAY] = start_dt.day
    return params


class SimpleCalendarAddEventWithLongDescription(calendar.SimpleCalendarAddOneEvent):
  """Add event with a very detailed description."""
  
  complexity = 3.8
  max_steps = 15
  template = (
      "In Simple Calendar Pro, create an event on {year}-{month}-{day} at {hour}h "
      "titled '{event_title}'. Set the description to: '{event_description}'."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    event = params[sqlite_validators.ROW_OBJECTS][0]
    
    long_desc = (
        "Agenda: 1. Review Q3 goals. 2. Discuss Q4 budget. "
        "3. Team building activity planning. 4. Miscellaneous."
    )
    event = dataclasses.replace(event, description=long_desc)
    
    params[sqlite_validators.ROW_OBJECTS] = [event]
    params["event_description"] = long_desc
    return params


class SimpleCalendarAddOverlappingEvent(calendar.SimpleCalendarAddOneEvent):
  """Add an event at the same time as an existing one (testing conflict handling/ignoring)."""
  
  complexity = 4.5
  max_steps = 15
  template = (
      "In Simple Calendar Pro, schedule '{event_title}' on {year}-{month}-{day} "
      "at {hour}h, even if there is another event at that time."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    target_event = params[sqlite_validators.ROW_OBJECTS][0]
    
    # Create a noise event that conflicts exactly
    conflict_event = dataclasses.replace(
        target_event,
        title="Existing Conflict",
        description="This event is already here."
    )
    
    # Add conflict to noise so it exists before task starts
    noise = params.get(sqlite_validators.NOISE_ROW_OBJECTS, [])
    noise.append(conflict_event)
    params[sqlite_validators.NOISE_ROW_OBJECTS] = noise
    
    return params


class SimpleCalendarDeleteAllFutureEvents(calendar.SimpleCalendarDeleteEvents):
  """Delete all events from a certain date onwards."""
  
  complexity = 5.0
  max_steps = 25
  n_rows = 5 # arbitrary target count
  template = (
      "In Simple Calendar Pro, delete all events scheduled on or after {year}-{month}-{day}."
  )
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Pivot date: Oct 20
    pivot_day = 20
    
    # Generate targets (>= Oct 20)
    targets = []
    for _ in range(cls.n_rows):
        day = random.randint(pivot_day, 30)
        targets.append(calendar.SimpleCalendarDeleteEvents._get_random_target_row(day))
        
    # Generate noise (< Oct 20)
    noise = []
    for _ in range(10):
        day = random.randint(1, pivot_day - 1)
        noise.append(calendar.SimpleCalendarDeleteEvents._get_random_target_row(day))

    return {
        "year": 2023, "month": 10, "day": pivot_day,
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise
    }


class SimpleCalendarAddWeeklyMeetingWithEnd(calendar.SimpleCalendarAddRepeatingEvent):
  """Add a weekly meeting but specify it lasts 1 hour."""
  
  complexity = 4.0
  max_steps = 15
  template = (
      "In Simple Calendar Pro, create a weekly event '{event_title}' on "
      "{day_of_week}s at {hour}h, lasting 1 hour."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Force duration 60 mins
    params = super().generate_random_params()
    target_event = params[sqlite_validators.ROW_OBJECTS][0]
    target_event = dataclasses.replace(target_event, end_ts=target_event.start_ts + 3600)
    
    params[sqlite_validators.ROW_OBJECTS] = [target_event]
    params[_DURATION_MINS] = 60
    params[_DAY_OF_WEEK] = target_event.start_datetime.strftime("%A")
    return params


class SimpleCalendarClearMonth(calendar.SimpleCalendarDeleteEvents):
  """Delete all events in October."""
  
  complexity = 5.0
  max_steps = 30
  template = "In Simple Calendar Pro, clear all events in October 2023."
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Targets: Random spread in October
    targets = calendar.generate_noise_events([], 15) # Generate 15 random events
    
    # Noise: None (since we clear everything), or maybe events in November if supported?
    # Environment is fixed to October 2023 context usually, but let's assume valid scope.
    
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: []
    }


class SimpleCalendarAddEventDurationHours(calendar.SimpleCalendarAddOneEvent):
  """Add event specifying duration in hours."""
  
  complexity = 3.6
  max_steps = 15
  template = (
      "In Simple Calendar Pro, create an event '{event_title}' on {year}-{month}-{day} "
      "at {hour}h that lasts for {duration_hours} hours."
  )
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    hours = random.randint(2, 5)
    target = params[sqlite_validators.ROW_OBJECTS][0]
    
    target = dataclasses.replace(
        target,
        end_ts=target.start_ts + (hours * 3600)
    )
    
    params[sqlite_validators.ROW_OBJECTS] = [target]
    params[_DURATION_HOURS] = hours
    return params


class SimpleCalendarDeleteEventByDescription(calendar.SimpleCalendarDeleteOneEvent):
  """Delete event with a specific description substring."""
  
  complexity = 4.2
  max_steps = 15
  template = (
      "In Simple Calendar Pro, delete the event on {year}-{month}-{day} that includes "
      "the description '{unique_desc}'."
  )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = super().generate_random_params()
    target = params[sqlite_validators.ROW_OBJECTS][0]
    unique_desc = "Secret Code: 12345"
    target = dataclasses.replace(target, description=f"Regular text. {unique_desc}")
    
    params[sqlite_validators.ROW_OBJECTS] = [target]
    params['unique_desc'] = unique_desc
    return params
