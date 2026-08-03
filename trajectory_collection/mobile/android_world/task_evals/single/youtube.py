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

"""Tasks for YouTube via Google Chrome."""

import random
import datetime
from typing import Any
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval

_APP_NAME = 'chrome'
_PACKAGE_NAME = 'com.android.chrome'

class _YouTube(task_eval.TaskEval):
    """YouTube (via Chrome) Base Class"""
    app_names = (_APP_NAME,)

    def initialize_task(self, env: interface.AsyncEnv) -> None:
        super().initialize_task(env)
        adb_utils.launch_app(_APP_NAME, env.controller)

    def tear_down(self, env: interface.AsyncEnv) -> None:
        super().tear_down(env)
        adb_utils.close_app(_APP_NAME, env.controller)

# --- 11 YouTube Tasks ---

class YouTubeManimStats(_YouTube):
    """Task 1: Search Manim tutorial and record statistics to Markor."""
    app_names = (_APP_NAME, "markor")
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Open Google Chrome and go to YouTube. Search for "Python Manim tutorial". Apply the search filters to restrict results to "This month" and sort by "View count". Find the top-ranked video, copy its share link, and the channel name. Then create a new file in Markor named Manim_Stats.md, paste these pieces of information into it, and save.'
    def is_successful(self, env): return 1.0 # The trajectory collection mainly relies on the process, and the judgment logic can be simplified
    @classmethod
    def generate_random_params(cls): return {}

class YouTubeCS229Notes(_YouTube):
    """Task 2: Find Stanford course chapters and record them to Markor."""
    app_names = (_APP_NAME, "markor")
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Open Google Chrome and go to YouTube. Search for "Stanford CS229 Lecture 1" and play the video. Expand the detailed description at the bottom of the video to find and open the Chapters list. Locate the chapter whose title contains "Supervised Learning" or similar keywords, and note down its starting timestamp. Open Markor, append a new line to the file Study_Notes.txt stating: "Supervised Learning starts at [the timestamp you found]".'
    def is_successful(self, env): return 1.0
    @classmethod
    def generate_random_params(cls): return {}

class YouTubeMARLTutorial(_YouTube):
    """Task 3: Find multi-agent reinforcement learning tutorials and save them to files with timestamps."""
    app_names = (_APP_NAME, "markor")
    complexity = 2.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Could you open Google Chrome and go to YouTube to look for a recent, beginner-friendly tutorial on multi-agent reinforcement learning? Once you find a good one, just grab the video title, the channel name, and the link. Then, create a new text file to jot these down. Oh, and please name the file something like MARL_tutorial_[current_timestamp].txt so I know exactly when we have it.'
    def is_successful(self, env): return 1.0
    @classmethod
    def generate_random_params(cls): return {}

class YouTubeCodingAssistantVibe(_YouTube):
    """Task 4: Summarize the comments of AI coding assistants."""
    app_names = (_APP_NAME, "markor")
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = "I'm really curious about what people are saying about the new AI coding assistants. Would you mind opening Google Chrome, going to YouTube, and searching for a recent comparison video? Skim through the top few comments and write up a quick summary of the general vibe. Just drop that summary into a markdown file, and stick a timestamp on the filename (like coding_assistants_vibe_[timestamp].md) to help me keep track of the versions."

    def is_successful(self, env): return 1.0
    @classmethod
    def generate_random_params(cls): return {}

class YouTubeLocalLLMHardware(_YouTube):
    """Task 5: YouTube check GPU recommendations + browser check prices -> Markor."""
    app_names = (_APP_NAME, "markor")
    complexity = 4.0
    schema = {'type': 'object', 'properties': {}}
    template = "Let's look into some new setups for running local LLMs. Open Google Chrome and go to YouTube to check out a hardware review to see what GPUs people are recommending right now, and then open a new tab in Chrome to check the official specs and pricing on the manufacturer's site. Throw the recommended specs and the price estimate into a markdown document named Local_LLM_Hardware_[timestamp].md"
    def is_successful(self, env): return 1.0
    @classmethod
    def generate_random_params(cls): return {}

class YouTubeNL2CodePlaylist(_YouTube):
    """Task 6: Filter short videos and create a private playlist."""
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = "I need to put together a quick study list for natural language to code generation. Could you open Google Chrome, go to YouTube, and search for 'NL2Code tutorials'? Filter the results to only show videos uploaded this year, and pick three videos that are strictly under 10 minutes long. Once you spot them, just create a new private playlist and add those three videos to it."

    def is_successful(self, env): return 1.0
    @classmethod
    def generate_random_params(cls): return {}

class YouTubeSaveAgentEvalPlaylist(_YouTube):
    """Task 7: Save agent evaluation playlist to the library."""
    complexity = 1.5
    schema = {'type': 'object', 'properties': {}}
    template = "I need to catch up on the latest agent evaluation methods for my research. Could you open Google Chrome, go to YouTube, and find a solid, up-to-date playlist covering this topic? Just save the whole thing to my library. Make sure it's actually a comprehensive playlist with a decent number of videos."

    def is_successful(self, env): return 1.0
    @classmethod
    def generate_random_params(cls): return {}

class YouTubeOpenHandsComplaints(_YouTube):
    """Task 8: Extract high-praise complaints from OpenHands tutorials."""
    complexity = 2.0
    schema = {'type': 'object', 'properties': {}}
    template = "I'm curious what kind of issues people are running into when setting up OpenHands or similar coding agents locally. Can you open Google Chrome, go to YouTube, and track down some popular, recent tutorials? Skim the comments to see what the most common complaints are. Just grab a couple of the most liked comments for me."

    def is_successful(self, env): return 1.0
    @classmethod
    def generate_random_params(cls): return {}

class YouTubeRioTravelGuide(_YouTube):
    """Task 9: Find the safety and transportation guide for Rio de Janeiro."""
    complexity = 2.0
    schema = {'type': 'object', 'properties': {}}
    template = "I'm heading to Rio de Janeiro for a conference soon and need to figure out the logistics. Could you open Google Chrome, go to YouTube, and find a highly-rated vlog or travel guide from the past year that specifically focuses on safety tips and getting around the city? Once you find a good one, just give the video link and the channel name to me."

    def is_successful(self, env): return 1.0
    @classmethod
    def generate_random_params(cls): return {}

class YouTubeTelescopeReview(_YouTube):
    """Task 10: Check the battery and alignment issues of the Celestron telescope."""
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = 'I\'m considering getting a beginner-friendly telescope for some stargazing in the mountains near Silicon Valley. Open Google Chrome and go to YouTube to look up a review for the \'Celestron NexStar 4SE\'. Find a video that\'s a year or two old, and check the comments specifically for the \'alignment process\' or \'battery life\'. I want to know the real-world frustrations before I buy.'
    
    def is_successful(self, env): return 1.0
    @classmethod
    def generate_random_params(cls): return {}

class YouTubeAutoGPTPortability(_YouTube):
    """Task 11: Compare two AutoGPT versions and recommend the most current one."""
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Open Google Chrome and go to YouTube. Find two different videos on \'AutoGPT setup\' or a similar framework. Compare their upload dates and the version numbers they mention in the titles. Just tell me which one seems to be the most \'current\' one I should follow, and why.'
    
    def is_successful(self, env): return 1.0
    @classmethod
    def generate_random_params(cls): return {}
