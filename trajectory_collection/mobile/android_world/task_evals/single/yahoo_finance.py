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

"""Tasks for the Yahoo Finance app."""

import random
from typing import Any
from android_world.env import adb_utils, interface
from android_world.task_evals import task_eval

_APP_NAME = 'yahoo finance' 
_PACKAGE_NAME = 'com.yahoo.mobile.client.android.finance'

class _YahooFinance(task_eval.TaskEval):
    """Yahoo Finance Base Class"""
    app_names = (_APP_NAME,)

    def initialize_task(self, env: interface.AsyncEnv) -> None:
        super().initialize_task(env)
        adb_utils.launch_app(_APP_NAME, env.controller)

    def tear_down(self, env: interface.AsyncEnv) -> None:
        super().tear_down(env)
        adb_utils.close_app(_APP_NAME, env.controller)

    def is_successful(self, env): return 1.0 # The trajectory collection mainly relies on the process, and the judgment logic can be simplified
    @classmethod
    def generate_random_params(cls): return {}

# --- 12 Yahoo Finance Tasks ---

class YahooSP500FiveYears(_YahooFinance):
    complexity = 2.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Open the S&P 500 stock and display the past 5 years price diagram.'

class YahooTopNews(_YahooFinance):
    complexity = 1.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Display the toppest new of the day.'

class YahooTeslaPriceAlert(_YahooFinance):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Create a price alert for the Tesla and set the target price to 30% off from current price.'

class YahooAddCircleWatchlist(_YahooFinance):
    complexity = 2.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Add CIRCLE to my watchlist.'

class YahooEditTeslaStats(_YahooFinance):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Edit the key statistics for Tesla, Add both "Day high" and "Day low".'

class YahooNVIDIAHistoricalChart(_YahooFinance):
    complexity = 4.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Search for NVIDIA (NVDA), navigate to its Historical Data section, change the time range to Max, switch the chart type to Line, and display the chart.'

class YahooTopGainersAnalysis(_YahooFinance):
    complexity = 4.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Find the Top Gainers in the U.S. market today, sort them by % change descending, and open the stock with the highest trading volume among the top 3.'

class YahooAppleComparison(_YahooFinance):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Search for Apple (AAPL), compare it with Microsoft (MSFT) and Google (GOOGL) using the Compare feature, and display their 1-year performance on the same chart.'

class YahooAmazonRevenueTrend(_YahooFinance):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Open Amazon (AMZN), navigate to Financials, switch to Quarterly view, and identify whether revenue has increased or decreased compared to the previous quarter.'

class YahooCreateTechPortfolio(_YahooFinance):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Create a new portfolio named “Tech Growth 2026”, add Tesla (TSLA), Meta (META), and NVIDIA (NVDA), and allocate equal virtual investment amounts of $10,000 to each stock.'

class YahooSPYRecurringAlert(_YahooFinance):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Set a recurring alert for SPY (S&P 500 ETF) to notify when the daily percentage change exceeds ±2% in either direction.'

class YahooCryptoWatchlist(_YahooFinance):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Navigate to the Crypto section, identify the cryptocurrency with the highest 24-hour trading volume, and add it to your watchlist.'


class YahooChipStocksResearch(_YahooFinance):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = "Search for NVIDIA (NVDA) on Yahoo Finance. Scroll down to find its current market cap and the P/E ratio (Trailing) in the 'Key Statistics' part. Next, search for AMD and find the same two values. Compare their P/E ratios and determine which one is higher. Finally, open Markor, create a new file named Chip_Stocks_Research.md, and list both companies with their market caps and P/E ratios, explicitly stating which company has the higher P/E ratio."


class YahooKOExDividendCalendar(_YahooFinance):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = "Search for Coca-Cola (KO) on Yahoo Finance. Scroll down to the 'Key Statistics' section and click on 'show all key statistics', then look for the 'Ex-Dividend Date' and remember the date. Then, open the Simple Calendar Pro app, navigate to that specific date in the calendar, and create a new event titled 'KO Ex-Dividend Date' with a description saying 'Check portfolio for dividend payment'."


class YahooBTCCryptoAction(_YahooFinance):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = "Navigate to the 'Crypto' section in the 'Markets' section of Yahoo Finance and find Bitcoin (BTC). Check its 24-hour percentage change. If the price has dropped by more than 2%, set an alarm in the Clock app for 1 hour from now with the label 'BTC Volatility Alert'. If it hasn't dropped that much, just add Bitcoin to your 'Watchlist' in Yahoo Finance instead."


class YahooAMZNEarningsDraft(_YahooFinance):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = "Open Amazon (AMZN) on Yahoo Finance and find the 'News' section. Locate the most recent article related to their quarterly earnings or financial performance. Skim the first few sentences of the article to find one key takeaway. Then, open Gmail, compose a new email to yourself with the subject 'AMZN Earnings Note', paste the key takeaway into the body, and save the email as a draft."


class YahooSP500VolatilityNote(_YahooFinance):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = "Go to the 'S&P 500' (GSPC) page on Yahoo Finance. Identify the 'Day High' and 'Day Low' prices. Calculate the difference between these two prices. Then, open Markor and find a folder named 'Finance'. Inside that folder, create a new document named SP500_Volatility.txt and record the Day High, Day Low, and the calculated difference you found."


class YahooAAPLSGDCost(_YahooFinance):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = "I'm thinking of buying some shares of Apple (AAPL). Look up its current stock price on Yahoo Finance. Then, pop open the Chrome browser to find the current exchange rate from USD to SGD (Singapore Dollar). Use the Calculator app to multiply the stock price by the exchange rate to find the cost in SGD. Finally, create a new note in Markor named AAPL_SGD_Cost.txt and record the final price in Singapore Dollars."


class YahooMSFTBuyPlan(_YahooFinance):
    complexity = 3.5
    schema = {'type': 'object', 'properties': {}}
    template = "Find the current trading price of Microsoft (MSFT) on Yahoo Finance. Next, open the Calculator app and calculate how many full shares you can afford with a total budget of $25,000 (divide 25,000 by the stock price and round down to the nearest whole number). After that, open Gmail and draft an email to yourself with the subject 'MSFT Buy Plan' stating: 'I can buy [number] shares with my $25,000 budget at the current price'."


class YahooNetflixCEOBio(_YahooFinance):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = "Go to the 'COMPANY PROFILE' tab of Netflix (NFLX) on Yahoo Finance and find the name of the current CEO listed under 'Key Executives'. Then, open the Chrome browser and search for this person's name to find their 'alma mater' (the college or university they graduated from). Finally, open Markor and append a line to a file named CEO_Bios.txt in this format: 'Netflix - [CEO Name] - [University Name]'."


class YahooTopGainerEmailDraft(_YahooFinance):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = "Go to Yahoo Finance and find the 'Top Gainers' in the U.S. market today. Identify the stock with the highest percentage gain and note its ticker symbol. Then, open Gmail and draft an email to yourself with the subject 'Market Watch: Top Gainer' and simply state in the body: 'The top gainer today is [Ticker Symbol]. Checked on April 15, 2026.'"


class YahooEditTeslaStatsKeepOrder(_YahooFinance):
    complexity = 3.0
    schema = {'type': 'object', 'properties': {}}
    template = 'Edit the key statistics for Tesla to add both "Day high" and "Day low". Do not remove or reorder any of the currently displayed statistics.'


class YahooAppleComparisonNoIndices(_YahooFinance):
    complexity = 4.5
    schema = {'type': 'object', 'properties': {}}
    template = 'Search for Apple (AAPL), compare it with Microsoft (MSFT) and Google (GOOGL) using the Compare feature, and display their 1-year performance on the same chart. Keep AAPL as the primary baseline asset, and do not include any broader market indices (like NASDAQ) in this specific comparison.'
