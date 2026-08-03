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

"""Additional tasks for the Markor app."""

import datetime
import random
from typing import Any

from absl import logging
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.single.markor import (
    Markor,
    _generate_random_note,
    _NOTE_TITLES,
)
from android_world.task_evals.utils import user_data_generation
from android_world.utils import datetime_utils
from android_world.utils import file_utils
from android_world.utils import fuzzy_match_lib


class MarkorRenameNote(Markor):
  """Task for renaming an existing note."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "original_name": {"type": "string"},
          "new_name": {"type": "string"},
      },
      "required": ["original_name", "new_name"],
  }
  template = "In Markor, rename the note {original_name} to {new_name}."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.content = user_data_generation.generate_random_string(20)
    file_utils.create_file(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.content,
    )
    user_data_generation.generate_noise_files(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Check original is gone
    if file_utils.check_file_or_folder_exists(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
    
    # Check new exists
    if not file_utils.check_file_or_folder_exists(
        self.params["new_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
      
    # Check content preserved
    return 1.0 if file_utils.check_file_content(
        file_utils.convert_to_posix_path(
            device_constants.MARKOR_DATA, self.params["new_name"]
        ),
        self.content,
        env.controller,
    ) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {
        "original_name": _generate_random_note().name,
        "new_name": _generate_random_note().name,
    }


class MarkorDuplicateNote(Markor):
  """Task for creating a copy of an existing note."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "source_name": {"type": "string"},
          "copy_name": {"type": "string"},
      },
      "required": ["source_name", "copy_name"],
  }
  template = (
      "Create a duplicate of the note {source_name} in Markor and name the"
      " copy {copy_name}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.content = user_data_generation.generate_random_string(20)
    file_utils.create_file(
        self.params["source_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.content,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Both must exist
    if not file_utils.check_file_or_folder_exists(
        self.params["source_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
      
    if not file_utils.check_file_or_folder_exists(
        self.params["copy_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0

    # Check content match
    return 1.0 if file_utils.check_file_content(
        file_utils.convert_to_posix_path(
            device_constants.MARKOR_DATA, self.params["copy_name"]
        ),
        self.content,
        env.controller,
    ) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {
        "source_name": _generate_random_note().name,
        "copy_name": _generate_random_note().name,
    }


class MarkorCreateShoppingList(Markor):
  """Task for creating a specific shopping list note."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "items": {"type": "string"},
      },
      "required": ["file_name", "items"],
  }
  template = (
      "Create a shopping list in Markor named {file_name} containing the"
      " following items: {items}."
  )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not file_utils.check_file_or_folder_exists(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0

    content = adb_utils.read_file(
        file_utils.convert_to_posix_path(
            device_constants.MARKOR_DATA, self.params["file_name"]
        ),
        env.controller
    )
    
    # Check if all items are present
    items_list = [i.strip() for i in self.params["items"].split(",")]
    return 1.0 if all(item in content for item in items_list) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    items = ["Milk", "Eggs", "Bread", "Cheese", "Apples", "Coffee"]
    selected_items = ", ".join(random.sample(items, 3))
    return {
        "file_name": "ShoppingList.md",
        "items": selected_items,
    }


class MarkorArchiveNote(Markor):
  """Task to move a note to an 'Archive' folder, creating the folder if needed."""

  complexity = 1.6
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
      },
      "required": ["file_name"],
  }
  template = "Move the note {file_name} to the 'Archive' folder in Markor."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_file(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content="content",
    )
    # Ensure Archive folder doesn't strictly need to exist yet, user might need to create it

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    archive_path = file_utils.convert_to_posix_path(
        device_constants.MARKOR_DATA, "Archive"
    )
    
    # Check file is NOT in root
    if file_utils.check_file_or_folder_exists(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
        return 0.0

    # Check file IS in Archive
    file_in_archive = file_utils.convert_to_posix_path(
        archive_path, self.params["file_name"]
    )
    # We need to list files in Archive to verify
    try:
        res = adb_utils.issue_generic_request(
            ["shell", "ls", archive_path], env.controller
        )
        files = res.generic.output.decode().splitlines()
        return 1.0 if self.params["file_name"] in files else 0.0
    except Exception:
        return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"file_name": _generate_random_note().name}


class MarkorCreateDailyLog(Markor):
  """Task to create a note named with the current date."""

  complexity = 1.2
  schema = {}
  template = (
      "Create a new note in Markor for a daily log. Name the file using today's"
      " date in the format YYYY-MM-DD.md."
  )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    today_str = datetime.datetime.now().strftime("%Y-%m-%d") + ".md"
    return 1.0 if file_utils.check_file_or_folder_exists(
        today_str,
        device_constants.MARKOR_DATA,
        env.controller,
    ) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class MarkorDeleteSpecificFolder(Markor):
  """Task to delete a specific folder."""

  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "folder_name": {"type": "string"},
      },
      "required": ["folder_name"],
  }
  template = "Delete the folder named {folder_name} in Markor."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    folder_path = file_utils.convert_to_posix_path(
        device_constants.MARKOR_DATA, self.params["folder_name"]
    )
    adb_utils.issue_generic_request(
        ["shell", "mkdir", "-p", folder_path], env.controller
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    exists = file_utils.check_file_or_folder_exists(
        self.params["folder_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    )
    return 0.0 if exists else 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"folder_name": "OldProjects"}


class MarkorAppendToNote(Markor):
  """Task to append text to the end of an existing note."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "append_text": {"type": "string"},
      },
      "required": ["file_name", "append_text"],
  }
  template = "Append the text \"{append_text}\" to the end of the note {file_name}."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.original_content = "Original content line."
    file_utils.create_file(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.original_content,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    content = adb_utils.read_file(
        file_utils.convert_to_posix_path(
            device_constants.MARKOR_DATA, self.params["file_name"]
        ),
        env.controller
    )
    # Robust check: The file should contain original content and end with appended text
    # Note: Markor might add newlines, so we use 'in' or strip()
    expected_end = self.params["append_text"]
    return 1.0 if self.original_content in content and expected_end in content else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {
        "file_name": _generate_random_note().name,
        "append_text": "This is the end.",
    }


class MarkorClearNote(Markor):
  """Task to clear all content from a note without deleting the file."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
      },
      "required": ["file_name"],
  }
  template = "Clear all text from the note {file_name}, leaving the file empty."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_file(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=user_data_generation.generate_random_string(50),
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not file_utils.check_file_or_folder_exists(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
      
    content = adb_utils.read_file(
        file_utils.convert_to_posix_path(
            device_constants.MARKOR_DATA, self.params["file_name"]
        ),
        env.controller
    )
    return 1.0 if not content.strip() else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"file_name": _generate_random_note().name}


class MarkorConvertTxtToMd(Markor):
  """Task to change a file extension from .txt to .md."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "base_name": {"type": "string"},
      },
      "required": ["base_name"],
  }
  template = (
      "Convert the file {base_name}.txt to Markdown format by renaming it to"
      " {base_name}.md."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_file(
        f"{self.params['base_name']}.txt",
        device_constants.MARKOR_DATA,
        env.controller,
        content="Some text",
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    txt_exists = file_utils.check_file_or_folder_exists(
        f"{self.params['base_name']}.txt",
        device_constants.MARKOR_DATA,
        env.controller,
    )
    md_exists = file_utils.check_file_or_folder_exists(
        f"{self.params['base_name']}.md",
        device_constants.MARKOR_DATA,
        env.controller,
    )
    return 1.0 if (not txt_exists and md_exists) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {"base_name": user_data_generation.generate_random_string(8)}


class MarkorDeleteOldestNote(Markor):
  """Task to delete the oldest note in the directory."""

  complexity = 1.4
  schema = {}
  template = "Delete the oldest note in Markor (the one modified longest ago)."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    # Create 3 notes with distinct timestamps
    self.notes = []
    for i in range(3):
        name = f"note_{i}.md"
        self.notes.append(name)
        file_utils.create_file(
            name,
            device_constants.MARKOR_DATA,
            env.controller,
            content=f"Content {i}"
        )
        # Space out modification times
        datetime_utils.advance_system_time(
          datetime.timedelta(minutes=10), env.controller
        )
    
    # The first one created is the oldest because we advanced time after each creation
    self.oldest = self.notes[0]
    self.others = self.notes[1:]

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Oldest should be gone
    if file_utils.check_file_or_folder_exists(
        self.oldest,
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
      
    # Others should exist
    for note in self.others:
        if not file_utils.check_file_or_folder_exists(
            note,
            device_constants.MARKOR_DATA,
            env.controller,
        ):
            return 0.0
            
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}