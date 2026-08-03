"""Additional complex tasks for Markor app."""

import datetime
import random
from typing import Any

from absl import logging
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.single import markor
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils
from android_world.utils import fuzzy_match_lib


class MarkorOrganizeByKeyword(markor.Markor):
  """Task to move files containing a specific keyword to a designated folder."""

  complexity = 3.0
  max_steps = 30
  schema = {
      "type": "object",
      "properties": {
          "keyword": {"type": "string"},
          "folder": {"type": "string"},
      },
      "required": ["keyword", "folder"],
  }
  template = (
      "Move all notes containing the word '{keyword}' to the folder '{folder}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.keyword = self.params["keyword"]
    self.folder = self.params["folder"]
    
    # Create destination folder
    file_utils.create_folder(self.folder, device_constants.MARKOR_DATA, env.controller)

    # Create files matching keyword
    self.target_files = []
    for i in range(3):
        name = f"target_{i}.md"
        content = f"This is a note about the {self.keyword}."
        file_utils.create_file(name, device_constants.MARKOR_DATA, env.controller, content=content)
        self.target_files.append(name)

    # Create noise files
    for i in range(3):
        name = f"noise_{i}.md"
        file_utils.create_file(name, device_constants.MARKOR_DATA, env.controller, content="Just random text.")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Check targets are in destination
    for name in self.target_files:
        path = f"{self.folder}/{name}"
        if not file_utils.check_file_or_folder_exists(path, device_constants.MARKOR_DATA, env.controller):
            return 0.0
        # Ensure they are NOT in root (Moved, not Copied)
        if file_utils.check_file_or_folder_exists(name, device_constants.MARKOR_DATA, env.controller):
            return 0.0
            
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "keyword": random.choice(["Project", "Urgent", "Recipe", "Idea"]),
        "folder": random.choice(["Work", "Personal", "Archive", "Important"])
    }


class MarkorArchiveByExtension(markor.Markor):
  """Task to move all files with a specific extension to an Archive folder."""

  complexity = 2.5
  max_steps = 25
  schema = {
      "type": "object",
      "properties": {
          "extension": {"type": "string"},
      },
      "required": ["extension"],
  }
  template = "Move all '{extension}' files to the 'Archive' folder."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.ext = self.params["extension"]
    file_utils.create_folder("Archive", device_constants.MARKOR_DATA, env.controller)

    self.targets = []
    # Create target files
    for i in range(3):
        name = f"doc_{i}{self.ext}"
        file_utils.create_file(name, device_constants.MARKOR_DATA, env.controller)
        self.targets.append(name)
    
    # Create noise files (different extension)
    other_ext = ".txt" if self.ext == ".md" else ".md"
    for i in range(3):
        file_utils.create_file(f"other_{i}{other_ext}", device_constants.MARKOR_DATA, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    for name in self.targets:
        # Check in archive
        if not file_utils.check_file_or_folder_exists(f"Archive/{name}", device_constants.MARKOR_DATA, env.controller):
            return 0.0
        # Check not in root
        if file_utils.check_file_or_folder_exists(name, device_constants.MARKOR_DATA, env.controller):
            return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {"extension": random.choice([".md", ".txt"])}


class MarkorCreateSummaryNote(markor.Markor):
  """Task to create a summary note containing the first line of other notes."""

  complexity = 4.0
  max_steps = 35
  schema = {
      "type": "object",
      "properties": {
          "f1": {"type": "string"},
          "f2": {"type": "string"},
          "f3": {"type": "string"},
      },
      "required": ["f1", "f2", "f3"],
  }
  template = (
      "Create a note named 'Summary.md'. It should contain the first line of "
      "'{f1}', '{f2}', and '{f3}', in that order."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.files = {}
    for i in range(1, 4):
        fname = f"note_{i}.md"
        line = user_data_generation.generate_random_string(15)
        content = f"{line}\n{user_data_generation.generate_random_string(20)}"
        file_utils.create_file(fname, device_constants.MARKOR_DATA, env.controller, content=content)
        self.files[f"f{i}"] = fname
        self.files[f"l{i}"] = line
    self.params.update(self.files)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    expected = f"{self.files['l1']}\n{self.files['l2']}\n{self.files['l3']}"
    return 1.0 if file_utils.check_file_content(
        file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, "Summary.md"),
        expected, env.controller
    ) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class MarkorBatchRenamePrefix(markor.Markor):
  """Task to rename multiple files by adding a prefix."""

  complexity = 3.0
  max_steps = 30
  schema = {
      "type": "object",
      "properties": {
          "f1": {"type": "string"},
          "f2": {"type": "string"},
          "prefix": {"type": "string"},
      },
      "required": ["f1", "f2", "prefix"],
  }
  template = "Rename '{f1}' and '{f2}' by adding the prefix '{prefix}_' to their names."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    for k in ['f1', 'f2']:
        file_utils.create_file(self.params[k], device_constants.MARKOR_DATA, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    prefix = self.params['prefix']
    for k in ['f1', 'f2']:
        old_name = self.params[k]
        new_name = f"{prefix}_{old_name}"
        if file_utils.check_file_or_folder_exists(old_name, device_constants.MARKOR_DATA, env.controller):
            return 0.0
        if not file_utils.check_file_or_folder_exists(new_name, device_constants.MARKOR_DATA, env.controller):
            return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "f1": markor._generate_random_note().name,
        "f2": markor._generate_random_note().name,
        "prefix": random.choice(["Draft", "Old", "Backup"])
    }


class MarkorConvertListToTasks(markor.Markor):
  """Task to convert a plain text list into a checkbox task list."""

  complexity = 3.5
  max_steps = 30
  schema = {
      "type": "object",
      "properties": {
          "src": {"type": "string"},
          "dst": {"type": "string"},
      },
      "required": ["src", "dst"],
  }
  template = (
      "Read the lines from '{src}' and create a new file '{dst}' where each "
      "line is converted into a checklist item (start with '- [ ] ')."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.items = [user_data_generation.generate_random_string(10) for _ in range(3)]
    content = "\n".join(self.items)
    file_utils.create_file(self.params['src'], device_constants.MARKOR_DATA, env.controller, content=content)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    expected = "\n".join([f"- [ ] {item}" for item in self.items])
    path = file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, self.params['dst'])
    
    # Check existence first
    if not file_utils.check_file_or_folder_exists(self.params['dst'], device_constants.MARKOR_DATA, env.controller):
        return 0.0

    res = adb_utils.issue_generic_request(["shell", "cat", path], env.controller)
    content = res.generic.output.decode().strip()
    return 1.0 if fuzzy_match_lib.fuzzy_match(content, expected) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "src": "ideas.txt",
        "dst": "tasks.md"
    }


class MarkorSplitNote(markor.Markor):
  """Task to split one note into two based on paragraphs."""

  complexity = 4.0
  max_steps = 35
  schema = {
      "type": "object",
      "properties": {
          "src": {"type": "string"},
          "dst1": {"type": "string"},
          "dst2": {"type": "string"},
      },
      "required": ["src", "dst1", "dst2"],
  }
  template = (
      "The note '{src}' has two paragraphs. Split it into two files: "
      "'{dst1}' containing the first paragraph, and '{dst2}' containing the second."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.p1 = user_data_generation.generate_random_string(20)
    self.p2 = user_data_generation.generate_random_string(20)
    content = f"{self.p1}\n\n{self.p2}"
    file_utils.create_file(self.params['src'], device_constants.MARKOR_DATA, env.controller, content=content)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    c1 = file_utils.check_file_content(
        file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, self.params['dst1']),
        self.p1, env.controller
    )
    c2 = file_utils.check_file_content(
        file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, self.params['dst2']),
        self.p2, env.controller
    )
    return 1.0 if c1 and c2 else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "src": "combined.md",
        "dst1": "part1.md",
        "dst2": "part2.md"
    }


class MarkorDeleteEmptyNotes(markor.Markor):
  """Task to find and delete empty notes."""

  complexity = 2.5
  max_steps = 25
  schema = {}
  template = "Find and delete all empty notes (files with no content) in the root directory."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.empty_files = [f"empty_{i}.md" for i in range(3)]
    for f in self.empty_files:
        file_utils.create_file(f, device_constants.MARKOR_DATA, env.controller, content="")
    
    # Noise
    for i in range(3):
        file_utils.create_file(f"full_{i}.md", device_constants.MARKOR_DATA, env.controller, content="Data")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    for f in self.empty_files:
        if file_utils.check_file_or_folder_exists(f, device_constants.MARKOR_DATA, env.controller):
            return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class MarkorSwapContent(markor.Markor):
  """Task to swap the contents of two files."""

  complexity = 3.5
  max_steps = 30
  schema = {
      "type": "object",
      "properties": {
          "f1": {"type": "string"},
          "f2": {"type": "string"},
      },
      "required": ["f1", "f2"],
  }
  template = "Swap the text contents of '{f1}' and '{f2}'."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.c1 = "Content A"
    self.c2 = "Content B"
    file_utils.create_file(self.params['f1'], device_constants.MARKOR_DATA, env.controller, content=self.c1)
    file_utils.create_file(self.params['f2'], device_constants.MARKOR_DATA, env.controller, content=self.c2)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    f1_ok = file_utils.check_file_content(
        file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, self.params['f1']),
        self.c2, env.controller
    )
    f2_ok = file_utils.check_file_content(
        file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, self.params['f2']),
        self.c1, env.controller
    )
    return 1.0 if f1_ok and f2_ok else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "f1": "FileA.md",
        "f2": "FileB.md"
    }


class MarkorCreateNestedFolder(markor.Markor):
  """Task to create a nested folder structure."""

  complexity = 1.5
  max_steps = 15
  schema = {
      "type": "object",
      "properties": {
          "parent": {"type": "string"},
          "child": {"type": "string"},
      },
      "required": ["parent", "child"],
  }
  template = "Create a folder structure '{parent}/{child}' in Markor."

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    path = f"{self.params['parent']}/{self.params['child']}"
    return 1.0 if file_utils.check_file_or_folder_exists(path, device_constants.MARKOR_DATA, env.controller) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "parent": "Documents",
        "child": "Invoices"
    }


class MarkorBackupNote(markor.Markor):
  """Task to copy a note to a backup folder."""

  complexity = 2.0
  max_steps = 20
  schema = {
      "type": "object",
      "properties": {
          "file": {"type": "string"},
          "folder": {"type": "string"},
      },
      "required": ["file", "folder"],
  }
  template = "Create a copy of '{file}' inside the folder '{folder}'."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.content = user_data_generation.generate_random_string(10)
    file_utils.create_file(self.params['file'], device_constants.MARKOR_DATA, env.controller, content=self.content)
    file_utils.create_folder(self.params['folder'], device_constants.MARKOR_DATA, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Original must exist
    if not file_utils.check_file_or_folder_exists(self.params['file'], device_constants.MARKOR_DATA, env.controller):
        return 0.0
    # Copy must exist with same content
    backup_path = f"{self.params['folder']}/{self.params['file']}"
    return 1.0 if file_utils.check_file_content(
        file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, backup_path),
        self.content, env.controller
    ) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "file": markor._generate_random_note().name,
        "folder": "Backup"
    }


class MarkorDeleteNotesContainingText(markor.Markor):
  """Task to delete notes containing a specific keyword."""

  complexity = 2.5
  max_steps = 25
  schema = {
      "type": "object",
      "properties": {
          "keyword": {"type": "string"},
      },
      "required": ["keyword"],
  }
  template = "Delete all notes that contain the text '{keyword}'."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.keyword = self.params['keyword']
    self.targets = []
    for i in range(3):
        name = f"del_{i}.md"
        file_utils.create_file(name, device_constants.MARKOR_DATA, env.controller, content=f"Text {self.keyword} Text")
        self.targets.append(name)
    
    # Noise
    for i in range(3):
        file_utils.create_file(f"keep_{i}.md", device_constants.MARKOR_DATA, env.controller, content="Safe text")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    for f in self.targets:
        if file_utils.check_file_or_folder_exists(f, device_constants.MARKOR_DATA, env.controller):
            return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {"keyword": "Obsolete"}


class MarkorDailyLogCreation(markor.Markor):
  """Task to create multiple specific daily log files."""

  complexity = 3.5
  max_steps = 30
  schema = {}
  template = (
      "Create three notes: 'Morning.md' with text 'Gym', 'Noon.md' with text "
      "'Lunch', and 'Evening.md' with text 'Read'."
  )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    checks = [("Morning.md", "Gym"), ("Noon.md", "Lunch"), ("Evening.md", "Read")]
    for name, content in checks:
        if not file_utils.check_file_content(
            file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, name),
            content, env.controller):
            return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}


class MarkorMoveDoneTasks(markor.Markor):
  """Task to move completed checklist items to a separate file."""

  complexity = 4.5
  max_steps = 40
  schema = {
      "type": "object",
      "properties": {
          "src": {"type": "string"},
          "dst": {"type": "string"},
      },
      "required": ["src", "dst"],
  }
  template = (
      "Cut the completed task lines (starting with '- [x]') from '{src}' and "
      "paste them into '{dst}'."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.open = "- [ ] Task 1"
    self.done = "- [x] Task 2"
    content = f"{self.open}\n{self.done}"
    file_utils.create_file(self.params['src'], device_constants.MARKOR_DATA, env.controller, content=content)
    file_utils.create_file(self.params['dst'], device_constants.MARKOR_DATA, env.controller, content="")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    src_content = adb_utils.issue_generic_request(
        ["shell", "cat", file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, self.params['src'])],
        env.controller).generic.output.decode()
    
    dst_content = adb_utils.issue_generic_request(
        ["shell", "cat", file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, self.params['dst'])],
        env.controller).generic.output.decode()

    # Src should NOT have done task
    if self.done in src_content: return 0.0
    # Dst SHOULD have done task
    if self.done not in dst_content: return 0.0
    
    # Check open task remains in src
    if self.open not in src_content: return 0.0
    
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {"src": "todo.md", "dst": "archive.md"}


class MarkorAppendDateToName(markor.Markor):
  """Task to rename a file by appending the date."""

  complexity = 2.0
  max_steps = 20
  schema = {
      "type": "object",
      "properties": {
          "file": {"type": "string"},
          "date": {"type": "string"},
      },
      "required": ["file", "date"],
  }
  template = "Rename '{file}' to '{file}_{date}' (e.g. if file is A.md and date is 2024, name it A_2024.md)."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_file(self.params['file'], device_constants.MARKOR_DATA, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    base, ext = self.params['file'].rsplit('.', 1)
    new_name = f"{base}_{self.params['date']}.{ext}"
    
    if file_utils.check_file_or_folder_exists(self.params['file'], device_constants.MARKOR_DATA, env.controller):
        return 0.0
    if not file_utils.check_file_or_folder_exists(new_name, device_constants.MARKOR_DATA, env.controller):
        return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "file": markor._generate_random_note().name,
        "date": "2024"
    }


class MarkorCreateIndex(markor.Markor):
  """Task to create an index file listing other files."""

  complexity = 3.5
  max_steps = 30
  schema = {
      "type": "object",
      "properties": {
          "f1": {"type": "string"},
          "f2": {"type": "string"},
      },
      "required": ["f1", "f2"],
  }
  template = "Create a note 'index.md' that lists the names of these files: '{f1}', '{f2}'."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_file(self.params['f1'], device_constants.MARKOR_DATA, env.controller)
    file_utils.create_file(self.params['f2'], device_constants.MARKOR_DATA, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    path = file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, "index.md")
    if not file_utils.check_file_or_folder_exists("index.md", device_constants.MARKOR_DATA, env.controller):
        return 0.0
        
    content = adb_utils.issue_generic_request(
        ["shell", "cat", path],
        env.controller).generic.output.decode()
    
    return 1.0 if self.params['f1'] in content and self.params['f2'] in content else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "f1": markor._generate_random_note().name,
        "f2": markor._generate_random_note().name
    }


class MarkorReplaceTextBatch(markor.Markor):
  """Task to find and replace text in multiple files."""

  complexity = 4.0
  max_steps = 40
  schema = {
      "type": "object",
      "properties": {
          "f1": {"type": "string"},
          "f2": {"type": "string"},
          "old": {"type": "string"},
          "new": {"type": "string"},
      },
      "required": ["f1", "f2", "old", "new"],
  }
  template = "In '{f1}' and '{f2}', replace the word '{old}' with '{new}'."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    content = f"This is {self.params['old']} version."
    for f in ['f1', 'f2']:
        file_utils.create_file(self.params[f], device_constants.MARKOR_DATA, env.controller, content=content)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    for f in ['f1', 'f2']:
        res = adb_utils.issue_generic_request(
            ["shell", "cat", file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, self.params[f])],
            env.controller).generic.output.decode()
        if self.params['new'] not in res or self.params['old'] in res:
            return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "f1": markor._generate_random_note().name,
        "f2": markor._generate_random_note().name,
        "old": "Draft",
        "new": "Final"
    }


class MarkorSeparateExtensionsTwoFolders(markor.Markor):
  """Task to separate files into folders by extension."""

  complexity = 3.5
  max_steps = 35
  schema = {
      "type": "object",
      "properties": {
          "f_md": {"type": "string"},
          "f_txt": {"type": "string"},
      },
      "required": ["f_md", "f_txt"],
  }
  template = "Move all .md files to '{f_md}' folder and all .txt files to '{f_txt}' folder."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_folder(self.params['f_md'], device_constants.MARKOR_DATA, env.controller)
    file_utils.create_folder(self.params['f_txt'], device_constants.MARKOR_DATA, env.controller)
    
    self.mds = ["a.md", "b.md"]
    self.txts = ["c.txt", "d.txt"]
    for f in self.mds + self.txts:
        file_utils.create_file(f, device_constants.MARKOR_DATA, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    for f in self.mds:
        if not file_utils.check_file_or_folder_exists(f"{self.params['f_md']}/{f}", device_constants.MARKOR_DATA, env.controller):
            return 0.0
        # Should not exist in root
        if file_utils.check_file_or_folder_exists(f, device_constants.MARKOR_DATA, env.controller): return 0.0
            
    for f in self.txts:
        if not file_utils.check_file_or_folder_exists(f"{self.params['f_txt']}/{f}", device_constants.MARKOR_DATA, env.controller):
            return 0.0
        if file_utils.check_file_or_folder_exists(f, device_constants.MARKOR_DATA, env.controller): return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {"f_md": "Markdown", "f_txt": "Text"}


class MarkorTemplateNote(markor.Markor):
  """Task to create a note matching a specific template."""

  complexity = 2.5
  max_steps = 25
  schema = {
      "type": "object",
      "properties": {
          "name": {"type": "string"},
          "title": {"type": "string"},
          "date": {"type": "string"},
          "tag": {"type": "string"},
      },
      "required": ["name", "title", "date", "tag"],
  }
  template = "Create a note '{name}' with the following 3 lines:\n# {title}\nDate: {date}\nTags: {tag}"

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    expected = f"# {self.params['title']}\nDate: {self.params['date']}\nTags: {self.params['tag']}"
    return 1.0 if file_utils.check_file_content(
        file_utils.convert_to_posix_path(device_constants.MARKOR_DATA, self.params['name']),
        expected, env.controller, exact_match=False # Allow fuzzy whitespace
    ) else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "name": "entry.md",
        "title": "My Journal",
        "date": "2024-01-01",
        "tag": "#daily"
    }


class MarkorRestoreFromTrash(markor.Markor):
  """Task to move a file from a subdirectory back to root."""

  complexity = 1.5
  max_steps = 15
  schema = {
      "type": "object",
      "properties": {
          "file": {"type": "string"},
          "folder": {"type": "string"},
      },
      "required": ["file", "folder"],
  }
  template = "Move the note '{file}' from the folder '{folder}' back to the main folder."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_folder(self.params['folder'], device_constants.MARKOR_DATA, env.controller)
    path = f"{self.params['folder']}/{self.params['file']}"
    # Create file in subfolder
    file_utils.create_file(path, device_constants.MARKOR_DATA, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Check present in root
    if not file_utils.check_file_or_folder_exists(self.params['file'], device_constants.MARKOR_DATA, env.controller):
        return 0.0
    # Check removed from trash
    path = f"{self.params['folder']}/{self.params['file']}"
    if file_utils.check_file_or_folder_exists(path, device_constants.MARKOR_DATA, env.controller):
        return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "file": markor._generate_random_note().name,
        "folder": "Trash"
    }


class MarkorCopyMultipleFiles(markor.Markor):
  """Task to copy multiple files to a folder."""

  complexity = 3.0
  max_steps = 30
  schema = {
      "type": "object",
      "properties": {
          "f1": {"type": "string"},
          "f2": {"type": "string"},
          "folder": {"type": "string"},
      },
      "required": ["f1", "f2", "folder"],
  }
  template = "Copy '{f1}' and '{f2}' to the folder '{folder}'."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_folder(self.params['folder'], device_constants.MARKOR_DATA, env.controller)
    file_utils.create_file(self.params['f1'], device_constants.MARKOR_DATA, env.controller)
    file_utils.create_file(self.params['f2'], device_constants.MARKOR_DATA, env.controller)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    # Check originals exist
    if not file_utils.check_file_or_folder_exists(self.params['f1'], device_constants.MARKOR_DATA, env.controller): return 0.0
    if not file_utils.check_file_or_folder_exists(self.params['f2'], device_constants.MARKOR_DATA, env.controller): return 0.0
    # Check copies exist
    if not file_utils.check_file_or_folder_exists(f"{self.params['folder']}/{self.params['f1']}", device_constants.MARKOR_DATA, env.controller): return 0.0
    if not file_utils.check_file_or_folder_exists(f"{self.params['folder']}/{self.params['f2']}", device_constants.MARKOR_DATA, env.controller): return 0.0
    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {
        "f1": "copy_me_1.md",
        "f2": "copy_me_2.md",
        "folder": "Copied"
    }