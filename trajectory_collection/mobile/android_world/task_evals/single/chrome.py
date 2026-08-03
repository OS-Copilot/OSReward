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

"""Tasks for the Chrome browser app."""

import random
from typing import Any
from android_world.env import adb_utils, interface
from android_world.task_evals import task_eval

_APP_NAME = 'google chrome' # Corresponds to the key in adb_utils

class _Chrome(task_eval.TaskEval):
    """Chrome Base Class"""
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

# --- 11 Chrome Tasks ---

class ChromePaperComparison(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Recently I came across several papers on GUI Agents—namely Step-GUI, MAI-UI, and UI-Venus. Please help me look up and compare the AndroidWorld benchmark results of their ~8B-parameter models. Note that the comparison should be based on Pass@1.'

class ChromeMobileWorldLeaderboard(_Chrome):
    app_names = (_APP_NAME, "markor")
    complexity = 4.0
    schema = {'type': 'object', 'properties': {}}
    template = 'I am a researcher in the GUI Agent field. Recently, a new benchmark called MobileWorld has become quite popular. Please check its leaderboard and compile the top three entries and their corresponding scores under all different settings. Create a new document named mobileworld.md in Marker to record this.'

class ChromeSaintPierreReservation(_Chrome):
    app_names = (_APP_NAME, "clock")
    complexity = 4.0
    schema = {'type': 'object', 'properties': {}}
    template = 'I will arrive in Singapore early on Friday the day after tomorrow for a trip, and I’d like to eat at a French restaurant called Saint Pierre. Please find its earliest opening time on Friday, then set an alarm for the day after tomorrow one hour in advance accordingly. The alarm description should be ‘Restaurant reservation’ to remind me.'

class ChromeAlibabaBenchmark(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'I am a researcher in the GUI Agent field. Recently, there was a fairly popular new benchmark, but I forgot its name. It is about dynamic evaluation of mobile agents and was released by Alibaba’s Multimodal Artificial Intelligence (MAI) team in late 2025. Please help me find it and tell me the benchmark’s name.'

class ChromeNationalGallerySG(_Chrome):
    complexity = 2.0
    schema = {'type': 'object', 'properties': {}}
    template = 'I’m traveling to Singapore on Friday. Could you please check whether the National Gallery Singapore is open tomorrow? What are its opening hours?'

class ChromePaperBookmarks(_Chrome):
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Please search in Chrome for the papers AndroidWorld and MobileWorld, and add both of their arXiv PDF links to my bookmarks.'

class ChromeSGHolidays(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'How many nationwide public holidays does Singapore have in 2026? Please verify this on the official government website.'

class ChromeCheapFlightNanjing(_Chrome):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = 'What is the cheapest direct flight ticket from Singapore to Nanjing for tomorrow, and which airline is it?'

class ChromeACL2026Calendar(_Chrome):
    app_names = (_APP_NAME, "simple calendar pro")
    complexity = 4.0
    schema = {'type': 'object', 'properties': {}}
    template = 'I’m planning to attend ACL 2026. Please check the conference dates. I plan to attend the main conference, so create a new calendar event for those main-conference days titled ‘Attend ACL Conference.’'

class ChromeNeurIPSDeadline(_Chrome):
    app_names = (_APP_NAME, "tasks")
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = 'I’m planning to submit to NeurIPS 2026. Please look up the abstract submission deadline, then create a reminder 7 days before the deadline to remind me.'

class ChromeBoonLayHotels(_Chrome):
    app_names = (_APP_NAME, "markor")
    complexity = 4.0
    schema = {'type': 'object', 'properties': {}}
    template = 'I live near Boon Lay MRT station in Singapore. My parents are coming to visit in April. Please help me find the nearest hotels rated 4 stars or above, and record the three closest ones and their prices in Marker by creating a new file named hotel.txt.'


class ChromeYouTubeOvercooked(_Chrome):
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = "Go to the YouTube website, search for videos related to the game 'Overcooked', and select one from the search results to play."


class ChromeBestBuyAmazonPhoneComparison(_Chrome):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Navigate to both BestBuy and Amazon to find the current models of the Samsung Galaxy Z Flip and the iPhone 17 Pro. Gather their technical specifications, current retail prices, and user ratings. Using this data, create a table comparing these three metrics and provide a summary identifying which smartphone offers better value for money.'


class ChromeYouTubeFlippedReviews(_Chrome):
    complexity = 4.0
    schema = {'type': 'object', 'properties': {}}
    template = "Go to YouTube website and search for 'Flipped movie reviews'. Filter or browse the results to find videos longer than 10 minutes, then identify and play the specific video among them that has the highest number of likes."


class ChromeCVPR2024MostCitedPaper(_Chrome):
    complexity = 5.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Identify the computer vision paper from CVPR 2024 that has the highest citation count on arXiv. For this paper, retrieve the full abstract, provide a list of all authors along with their respective institutional affiliations, and generate its complete BibTeX citation entry.'


class ChromeNipahVirusUpdates(_Chrome):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Conduct a search for the most recent updates and information regarding the Nipah virus. Synthesize the collected data into a comprehensive report and organize the key events into a chronological timeline.'


class ChromeTrendingMoviesWeekComparison(_Chrome):
    complexity = 5.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Go to IMDb and Rotten Tomatoes to find the top 5 trending movies for the specific week of October 27 to November 2, 2025. Record the genres and ratings for each of these 5 films from both sites, then calculate and state which movie has the highest average rating across the two platforms.'


class ChromeHKUNLPLatestSeminar(_Chrome):
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Go to the HKUNLP Group site and tell me when the latest student seminar took place.'


class ChromeCVPRLocations2023To2025(_Chrome):
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Go to the CVF official site and list the locations where CVPR was held from 2023 to 2025.'


class ChromeHKUCDSFacultyQS(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Go to the HKU CDS homepage to find the faculty size and their QS World Ranking in DS & AI.'


class ChromeFudanNLPBooksCount(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Find the total number of books that have been authored by the FudanNLP group.'


class ChromeAppleAirPodsLowestPrice(_Chrome):
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Provide the lowest price currently listed for a pair of AirPods on the official Apple website.'


class ChromeArxivLLMPdfsJanuary2026(_Chrome):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Locate computer science papers published in January 2026 on arXiv that contain "large language model" in their titles, and download the corresponding PDF files.'


class ChromeRedAndBlackWikipedia(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Go to Wikipedia, provide a biographical summary of the author of the novel The Red and the Black and list the various adaptations of this work.'


class ChromeQS2026Top100Changes(_Chrome):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Identify which universities newly entered or dropped out of the top 100 in the 2026 QS World University Rankings compared to 2025, and determine which continents these changes are primarily located in.'


class ChromePhotosynthesisDefinitionComparison(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Compare the definitions of "photosynthesis" provided by Wikipedia and Britannica, and summarize their similarities and differences in a single sentence.'


class ChromeVSCodeBugIssues(_Chrome):
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Navigate to the "Issues" tab of the microsoft/vscode repository and determine the current number of open issues tagged with the "bug" label.'


class ChromeWTOYouTubePlaylists(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Go to the YouTube website and locate the official World Trade Organization (WTO) YouTube channel, then list the titles of the playlists that contain the most videos.'


class ChromeOpenAIMultiAgentResearchCount(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Go to the Research page on OpenAI\'s official website and count the total number of research articles categorized under the "multi-agent" topic.'


class ChromeF12026ReserveDriver(_Chrome):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = 'For the F1 team making its debut in the 2026 season, identify the citizenship of their designated reserve driver.'


class ChromeF1PointsImprovement2025(_Chrome):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = 'How many F1 teams managed to improve their total points tally in 2025 compared to their performance in 2024?'


class ChromeACL2024KanzhiChengPages(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Determine the starting and ending page numbers for the paper featured in ACL 2024 where Kanzhi Cheng is listed as the primary author.'


class ChromeHighestWeeklyCovidCases(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'What is the highest number of COVID-19 infections ever documented within a single seven-day period?'


class ChromeWTOAnnualReport2025(_Chrome):
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Locate and provide the 2025 edition of the World Trade Organization (WTO) Annual Report.'


class ChromeBoonLayHotelsExcludeHostels(_Chrome):
    app_names = (_APP_NAME, "markor")
    complexity = 4.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Please help me find the nearest hotels rated 4 stars or above, and record the three closest ones and their prices in Marker by creating a new file named hotel.txt. Do not include any hostels, capsule hotels, or Airbnb rentals in your search.'


class ChromeNeurIPSDeadlineAbstractOnly(_Chrome):
    app_names = (_APP_NAME, "tasks")
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Please look up the submission deadline of NeurIPS 2026, then create a reminder 7 days before the deadline to remind me. Make sure you use the Abstract Deadline, not the full paper deadline.'


class ChromeCheapFlightNanjingDirectOnly(_Chrome):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = 'What is the cheapest direct flight ticket from Singapore to Nanjing for tomorrow, and which airline is it? Ensure you only look at direct flights with zero layovers.'


class ChromeSGHolidaysOfficialMOMOnly(_Chrome):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'How many nationwide public holidays does Singapore have in 2026? Please verify this on the official MOM government website. Do not count school holidays or unofficial cultural observances.'


class ChromePaperBookmarksDirectPdfOnly(_Chrome):
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Please search in Chrome for the papers AndroidWorld and MobileWorld, and add both of their arXiv PDF links to my bookmarks. Make sure you bookmark the direct PDF URLs, not the arXiv abstract. Do not create any new bookmark folders for them.'
