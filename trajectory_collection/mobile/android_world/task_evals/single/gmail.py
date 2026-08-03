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

"""Tasks for the Gmail app."""

import random
from typing import Any
from android_world.env import adb_utils, interface
from android_world.task_evals import task_eval

_APP_NAME = 'gmail' 
_PACKAGE_NAME = 'com.google.android.gm'

class _Gmail(task_eval.TaskEval):
    """Gmail Base Class"""
    app_names = (_APP_NAME,)

    def initialize_task(self, env: interface.AsyncEnv) -> None:
        super().initialize_task(env)
        adb_utils.launch_app(_APP_NAME, env.controller)

    def tear_down(self, env: interface.AsyncEnv) -> None:
        super().tear_down(env)
        adb_utils.close_app(_APP_NAME, env.controller)

    def is_successful(self, env): return 1.0 
    @classmethod
    def generate_random_params(cls): return {}

# --- Additional 7 Complex Gmail Tasks ---

class GmailTripSummary(_Gmail):
    complexity = 5.0
    schema = {'type': 'object', 'properties': {}}
    template = "Search my Gmail for the most recent email with the subject 'Flight Confirmation'. Locate the flight number, departure gate, and the scheduled boarding time in the body. Then, search for another email titled 'Hotel Booking'. Once you have both, draft a new email to myself with the subject 'Trip Summary: [Destination Name]', organizing the flight and hotel details into a clean bulleted list."

class GmailCompareProjectStatus(_Gmail):
    complexity = 4.0
    schema = {'type': 'object', 'properties': {}}
    template = "Find the two most recent emails with the subject 'Project Alpha Status'. Compare the 'Completion Percentage' mentioned in both. Identify which email has the higher percentage (the most recent update), and mark it as starred."

class GmailHarvestGithubLinks(_Gmail):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = "Search for the latest newsletter from 'AI Weekly' or 'Tech Dispatch'. Scan the body for any URLs that point to 'github.com'. Extract the titles of the projects associated with those links. Then, create a new Gmail draft titled 'GitHub Resources to Check', pasting the project names and their respective links into the body."

class GmailDailyExpenseReport(_Gmail):
    complexity = 5.0
    schema = {'type': 'object', 'properties': {}}
    template = "Locate all emails received today that contain keywords like 'Receipt', 'Invoice', or 'Amount Paid'. For each email found, extract the merchant name and the total dollar amount. Send a new email to 'czd.alex@outlook.com' with the subject 'Daily Expense Log', presenting the data in a professional table format."

class GmailMeetingConflictResolver(_Gmail):
    complexity = 5.0
    schema = {'type': 'object', 'properties': {}}
    template = "Find a recent email with 'Meeting' or 'gathering' in the subject. Extract the date and time of the event. Then, search my 'Primary' tab for any other emails that also have 'Meeting' or 'gathering' in the subject but mention a different event at that same time. If a conflict exists, reply to the latest invite saying: 'I might have a conflict, let me double-check'."

class GmailSentFollowUp(_Gmail):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = "Go to my Sent folder and locate the most recent email I sent with the subject 'Partnership Inquiry' or 'Job Application'. Check the message thread to see if there has been any reply from the recipient. If the last message in the thread is still my original sent email, draft a polite follow-up reply in that same thread. In the draft, mention that I am 'just checking in to see if they've had a chance to review my previous note.' ."

class GmailReportSpamAction(_Gmail):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = "Scan my 'Primary' inbox for any email with a highly suspicious subject line, such as 'You won a lottery', 'Inheritance', or 'Urgent: Account Locked'. Open the most recent one, click the 'More' icon (three vertical dots) in the top-right corner of the email and select 'Report spam'."