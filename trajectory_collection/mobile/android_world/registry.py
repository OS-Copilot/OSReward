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

"""Registers the task classes."""

import os
import types
from typing import Any, Final

from android_world.task_evals import task_eval
from android_world.task_evals.single import clock
from android_world.task_evals.single import expense
from android_world.task_evals.single import chrome
from android_world.task_evals.single import gmail
from android_world.task_evals.single import google_maps
from android_world.task_evals.single import markor
from android_world.task_evals.single import recipe
from android_world.task_evals.single import retro_music
from android_world.task_evals.single import yahoo_finance
from android_world.task_evals.single import youtube
from android_world.task_evals.single.calendar import calendar
from android_world.task_evals.single.extensions import extensions_calendar as calendar_ext
from android_world.task_evals.single.extensions import extensions_clock as clock_ext
from android_world.task_evals.single.extensions import extensions_expense as expense_ext
from android_world.task_evals.single.extensions import extensions_markor as markor_ext
from android_world.task_evals.single.extensions import extensions_recipe as recipe_ext
from android_world.task_evals.single.extensions import extensions_retro_music as retro_music_ext
from android_world.task_evals.single.extensions import extensions_settings as settings_ext


def get_families() -> list[str]:
  return [
      TaskRegistry.ANDROID_WORLD_EXT_FAMILY,
  ]


class TaskRegistry:
  """Registry of the OSReward extension tasks."""

  ANDROID_WORLD_EXT_FAMILY: Final[str] = 'android_world_extension'

  # Which task set the `android_world_extension` family serves. Override with
  # the EXT_SUITE environment variable; see `__init__`.
  #   'aw'      -> _EXT_AW_TASKS       131 verified AndroidWorld extension tasks
  #   'new_app' -> _EXT_NEW_APP_TASKS  115 new-app tasks (phase 1 + 2 + 2-neg)
  EXT_SUITE: Final[str] = os.environ.get('EXT_SUITE', 'aw')

  # Maps each task name to its class, to construct instances of a task.
  ANDROID_EXT_TASK_REGISTRY = {}

  def get_registry(self, family: str) -> Any:
    """Gets the task registry for the given family.

    Args:
      family: The family.

    Returns:
      Task registry.

    Raises:
      ValueError: If provided family doesn't exist.
    """
    if family == self.ANDROID_WORLD_EXT_FAMILY:
      return self.ANDROID_EXT_TASK_REGISTRY
    else:
      raise ValueError(f'Unsupported family: {family}')

  # Phase 1 tasks for the apps added on top of the base AndroidWorld
  # suite. All are runnable.
  #
  # Per app: Chrome 11, Gmail 7, Google Maps 12, Yahoo Finance 12, YouTube 11  =>  53 total
  _EXT_NEW_APP_PHASE1_TASKS = (
    # ------------------------------------------------------------------- Chrome
    chrome.ChromeACL2026Calendar,
    chrome.ChromeAlibabaBenchmark,
    chrome.ChromeBoonLayHotels,
    chrome.ChromeCheapFlightNanjing,
    chrome.ChromeMobileWorldLeaderboard,
    chrome.ChromeNationalGallerySG,
    chrome.ChromeNeurIPSDeadline,
    chrome.ChromePaperBookmarks,
    chrome.ChromePaperComparison,
    chrome.ChromeSGHolidays,
    chrome.ChromeSaintPierreReservation,

    # -------------------------------------------------------------------- Gmail
    gmail.GmailCompareProjectStatus,
    gmail.GmailDailyExpenseReport,
    gmail.GmailHarvestGithubLinks,
    gmail.GmailMeetingConflictResolver,
    gmail.GmailReportSpamAction,
    gmail.GmailSentFollowUp,
    gmail.GmailTripSummary,

    # -------------------------------------------------------------- Google Maps
    google_maps.GoogleMapAccessibleAttractions,
    google_maps.GoogleMapCheapestHighRatedHotel,
    google_maps.GoogleMapDownloadOffline,
    google_maps.GoogleMapFindHighRatedLateNightRestaurant,
    google_maps.GoogleMapFoodSearchHistory,
    google_maps.GoogleMapHospitalAndGasRoute,
    google_maps.GoogleMapHotelPriceRange,
    google_maps.GoogleMapNavigateToLocation,
    google_maps.GoogleMapParksPublicTransit,
    google_maps.GoogleMapSaveTopBars,
    google_maps.GoogleMapSearchLocation,
    google_maps.GoogleMapShortestHistoryDistance,

    # ------------------------------------------------------------ Yahoo Finance
    yahoo_finance.YahooAddCircleWatchlist,
    yahoo_finance.YahooAmazonRevenueTrend,
    yahoo_finance.YahooAppleComparison,
    yahoo_finance.YahooCreateTechPortfolio,
    yahoo_finance.YahooCryptoWatchlist,
    yahoo_finance.YahooEditTeslaStats,
    yahoo_finance.YahooNVIDIAHistoricalChart,
    yahoo_finance.YahooSP500FiveYears,
    yahoo_finance.YahooSPYRecurringAlert,
    yahoo_finance.YahooTeslaPriceAlert,
    yahoo_finance.YahooTopGainersAnalysis,
    yahoo_finance.YahooTopNews,

    # ------------------------------------------------------------------ YouTube
    youtube.YouTubeAutoGPTPortability,
    youtube.YouTubeCS229Notes,
    youtube.YouTubeCodingAssistantVibe,
    youtube.YouTubeLocalLLMHardware,
    youtube.YouTubeMARLTutorial,
    youtube.YouTubeManimStats,
    youtube.YouTubeNL2CodePlaylist,
    youtube.YouTubeOpenHandsComplaints,
    youtube.YouTubeRioTravelGuide,
    youtube.YouTubeSaveAgentEvalPlaylist,
    youtube.YouTubeTelescopeReview,
  )

  # Phase 2 tasks for the new apps: cross-app flows that chain a new app
  # with a base AndroidWorld app. All are runnable.
  #
  # Per app: Calendar 6, Chrome 23, Expense 6, Google Maps 5, Retro Music 5, Yahoo Finance 9  =>  54 total
  _EXT_NEW_APP_PHASE2_TASKS = (
    # ----------------------------------------------------------------- Calendar
    calendar.SimpleCalendarDeepWorkSlot,
    calendar.SimpleCalendarFlightFromGmail,
    calendar.SimpleCalendarGymSessionTomorrow,
    calendar.SimpleCalendarMoveProjectMeeting,
    calendar.SimpleCalendarMuseumVisitFromMaps,
    calendar.SimpleCalendarWeekendPlanToMarkor,

    # ------------------------------------------------------------------- Chrome
    chrome.ChromeACL2024KanzhiChengPages,
    chrome.ChromeAppleAirPodsLowestPrice,
    chrome.ChromeArxivLLMPdfsJanuary2026,
    chrome.ChromeBestBuyAmazonPhoneComparison,
    chrome.ChromeCVPR2024MostCitedPaper,
    chrome.ChromeCVPRLocations2023To2025,
    chrome.ChromeF12026ReserveDriver,
    chrome.ChromeF1PointsImprovement2025,
    chrome.ChromeFudanNLPBooksCount,
    chrome.ChromeHKUCDSFacultyQS,
    chrome.ChromeHKUNLPLatestSeminar,
    chrome.ChromeHighestWeeklyCovidCases,
    chrome.ChromeNipahVirusUpdates,
    chrome.ChromeOpenAIMultiAgentResearchCount,
    chrome.ChromePhotosynthesisDefinitionComparison,
    chrome.ChromeQS2026Top100Changes,
    chrome.ChromeRedAndBlackWikipedia,
    chrome.ChromeTrendingMoviesWeekComparison,
    chrome.ChromeVSCodeBugIssues,
    chrome.ChromeWTOAnnualReport2025,
    chrome.ChromeWTOYouTubePlaylists,
    chrome.ChromeYouTubeFlippedReviews,
    chrome.ChromeYouTubeOvercooked,

    # ------------------------------------------------------------------ Expense
    expense.ExpenseAddChickenRice,
    expense.ExpenseBudgetReviewAlarm,
    expense.ExpenseDailySpendToMarkor,
    expense.ExpenseLogBankSmsSocial,
    expense.ExpenseNetflixReviewTask,
    expense.ExpenseUsdToSgdEntertainment,

    # -------------------------------------------------------------- Google Maps
    google_maps.GoogleMapGardensTripCalendar,
    google_maps.GoogleMapLauPaSatDinnerPlan,
    google_maps.GoogleMapOrchardClinicAlarm,
    google_maps.GoogleMapSingaporeDistanceCalc,
    google_maps.GoogleMapVivoCityShoppingEmail,

    # -------------------------------------------------------------- Retro Music
    retro_music.RetroFavoritesToMarkor,
    retro_music.RetroJustBlackTheme,
    retro_music.RetroMorningVibePlaylist,
    retro_music.RetroRecentSongSms,
    retro_music.RetroRepeatCityOfStars,

    # ------------------------------------------------------------ Yahoo Finance
    yahoo_finance.YahooAAPLSGDCost,
    yahoo_finance.YahooAMZNEarningsDraft,
    yahoo_finance.YahooBTCCryptoAction,
    yahoo_finance.YahooChipStocksResearch,
    yahoo_finance.YahooKOExDividendCalendar,
    yahoo_finance.YahooMSFTBuyPlan,
    yahoo_finance.YahooNetflixCEOBio,
    yahoo_finance.YahooSP500VolatilityNote,
    yahoo_finance.YahooTopGainerEmailDraft,
  )

  # Phase 2 negative-constraint variants: same flows as above, with an
  # exclusion the agent must honour. All are runnable.
  #
  # Per app: Calendar 1, Chrome 5, Yahoo Finance 2  =>  8 total
  _EXT_NEW_APP_PHASE2_NEG_TASKS = (
    # ----------------------------------------------------------------- Calendar
    calendar.SimpleCalendarACLMainConferenceOnly,

    # ------------------------------------------------------------------- Chrome
    chrome.ChromeBoonLayHotelsExcludeHostels,
    chrome.ChromeCheapFlightNanjingDirectOnly,
    chrome.ChromeNeurIPSDeadlineAbstractOnly,
    chrome.ChromePaperBookmarksDirectPdfOnly,
    chrome.ChromeSGHolidaysOfficialMOMOnly,

    # ------------------------------------------------------------ Yahoo Finance
    yahoo_finance.YahooAppleComparisonNoIndices,
    yahoo_finance.YahooEditTeslaStatsKeepOrder,
  )

  # All new-app tasks in one list: phase 1, phase 2 and the phase 2 negative
  # variants. 53 + 54 + 8 = 115 tasks, all runnable.
  _EXT_NEW_APP_TASKS = (
      _EXT_NEW_APP_PHASE1_TASKS
      + _EXT_NEW_APP_PHASE2_TASKS
      + _EXT_NEW_APP_PHASE2_NEG_TASKS
  )


  # The verified extension suite: every task below has at least one complete
  # human-annotated trial, so it is known to be reachable and gradeable on the
  # emulator. This is the tuple used for the `android_world_extension` family.
  #
  # Tasks that never produced a complete trial are kept as comments at the end
  # of their app cluster, so the full authored set stays visible for reference.
  #
  # Per app (registered / authored):
  #   Calendar 22/24, Clock 9/10, Expense 24/27,
  #   Markor 22/30, Recipe 28/29, Retro Music 26/28  =>  131 / 148
  _EXT_AW_TASKS = (
    # ---------------------------------------------------------------- Calendar
    calendar_ext.SimpleCalendarAddAllDayEvent,
    calendar_ext.SimpleCalendarAddEveningEvent,
    calendar_ext.SimpleCalendarAddEventDurationHours,
    calendar_ext.SimpleCalendarAddEventWithLocation,
    calendar_ext.SimpleCalendarAddEventWithLongDescription,
    calendar_ext.SimpleCalendarAddEventWithStartEnd,
    calendar_ext.SimpleCalendarAddLongEvent,
    calendar_ext.SimpleCalendarAddMorningEvent,
    calendar_ext.SimpleCalendarAddOneEventNextWeek,
    calendar_ext.SimpleCalendarAddOneEventThisWeekend,
    calendar_ext.SimpleCalendarAddOneEventToday,
    calendar_ext.SimpleCalendarAddOverlappingEvent,
    calendar_ext.SimpleCalendarAddThreeEventsDifferentDays,
    calendar_ext.SimpleCalendarAddTwoEventsSameDay,
    calendar_ext.SimpleCalendarAddWeeklyMeetingWithEnd,
    calendar_ext.SimpleCalendarDeleteAllEventsInWeek,
    calendar_ext.SimpleCalendarDeleteEventByDescription,
    calendar_ext.SimpleCalendarDeleteEventByLocation,
    calendar_ext.SimpleCalendarDeleteEventByTitle,
    calendar_ext.SimpleCalendarDeleteEventsNextWeek,
    calendar_ext.SimpleCalendarDeleteEventsThisWeekend,
    calendar_ext.SimpleCalendarDeleteEventsTomorrow,
    # No complete trial:
    # calendar_ext.SimpleCalendarClearMonth,
    # calendar_ext.SimpleCalendarDeleteAllFutureEvents,

    # ------------------------------------------------------------------- Clock
    clock_ext.ClockTimerEntryComplex,
    clock_ext.ClockTimerEntryFifteenMinutes,
    clock_ext.ClockTimerEntryFiveMinutes,
    clock_ext.ClockTimerEntryFortyFiveSeconds,
    clock_ext.ClockTimerEntryMaxTime,
    clock_ext.ClockTimerEntryOneHour,
    clock_ext.ClockTimerEntryOneHourThirtyMinutes,
    clock_ext.ClockTimerEntryPreciseCooking,
    clock_ext.ClockTimerEntryThirtyMinutes,
    # No complete trial:
    # clock_ext.ClockTimerEntryLongDuration,

    # ----------------------------------------------------------------- Expense
    expense_ext.ExpenseAddBackdated,
    expense_ext.ExpenseAddBudgetCorrection,
    expense_ext.ExpenseAddBusinessTrip,
    expense_ext.ExpenseAddDetailed,
    expense_ext.ExpenseAddHighValueSingle,
    expense_ext.ExpenseAddMonthlyBills,
    expense_ext.ExpenseAddMonthlySubscriptions,
    expense_ext.ExpenseAddMultipleFoodCategory,
    expense_ext.ExpenseAddPartyExpenses,
    expense_ext.ExpenseAddSpecificDate,
    expense_ext.ExpenseAddWithSpecificNote,
    expense_ext.ExpenseCleanupMisc,
    expense_ext.ExpenseDeduplicateStrict,
    expense_ext.ExpenseDeleteAllExceptFood,
    expense_ext.ExpenseDeleteAllIncome,
    expense_ext.ExpenseDeleteByDate,
    expense_ext.ExpenseDeleteByNameSubstring,
    expense_ext.ExpenseDeleteEntertainmentAndSocial,
    expense_ext.ExpenseDeleteFood,
    expense_ext.ExpenseDeleteHighValueExpenses,
    expense_ext.ExpenseDeleteHousing,
    expense_ext.ExpenseDeleteLowValue,
    expense_ext.ExpenseDeletePaidByCard,
    expense_ext.ExpenseDeleteUrgent,
    # No complete trial:
    # expense_ext.ExpenseAddVacationExpenses,
    # expense_ext.ExpenseDeleteSmallExpenses,
    # expense_ext.ExpenseDeleteSpecificNote,

    # ------------------------------------------------------------------ Markor
    markor_ext.MarkorAppendDateToName,
    markor_ext.MarkorArchiveNote,
    markor_ext.MarkorBatchRenamePrefix,
    markor_ext.MarkorConvertListToTasks,
    markor_ext.MarkorConvertTxtToMd,
    markor_ext.MarkorCreateDailyLog,
    markor_ext.MarkorCreateIndex,
    markor_ext.MarkorCreateNestedFolder,
    markor_ext.MarkorCreateShoppingList,
    markor_ext.MarkorCreateSummaryNote,
    markor_ext.MarkorDailyLogCreation,
    markor_ext.MarkorDeleteEmptyNotes,
    markor_ext.MarkorDeleteNotesContainingText,
    markor_ext.MarkorDeleteOldestNote,
    markor_ext.MarkorDeleteSpecificFolder,
    markor_ext.MarkorDuplicateNote,
    markor_ext.MarkorMoveDoneTasks,
    markor_ext.MarkorRenameNote,
    markor_ext.MarkorReplaceTextBatch,
    markor_ext.MarkorSplitNote,
    markor_ext.MarkorSwapContent,
    markor_ext.MarkorTemplateNote,
    # No complete trial:
    # markor_ext.MarkorAppendToNote,
    # markor_ext.MarkorArchiveByExtension,
    # markor_ext.MarkorBackupNote,
    # markor_ext.MarkorClearNote,
    # markor_ext.MarkorCopyMultipleFiles,
    # markor_ext.MarkorOrganizeByKeyword,
    # markor_ext.MarkorRestoreFromTrash,
    # markor_ext.MarkorSeparateExtensionsTwoFolders,

    # ------------------------------------------------------------------ Recipe
    recipe_ext.RecipeAddDetailed,
    recipe_ext.RecipeAddFavoriteRecipe,
    recipe_ext.RecipeAddHighProteinRecipe,
    recipe_ext.RecipeAddLargeBatch,
    recipe_ext.RecipeAddMarkorByIngredient,
    recipe_ext.RecipeAddMarkorFavorites,
    recipe_ext.RecipeAddMarkorHealthy,
    recipe_ext.RecipeAddMultipleRecipesFromMarkorQuick,
    recipe_ext.RecipeAddPartyRecipe,
    recipe_ext.RecipeAddQuickMeal,
    recipe_ext.RecipeAddSourceRecipe,
    recipe_ext.RecipeDeleteByDescriptionKeyword,
    recipe_ext.RecipeDeleteByPrepTime,
    recipe_ext.RecipeDeleteByServings,
    recipe_ext.RecipeDeleteBySource,
    recipe_ext.RecipeDeleteByTitleSubstring,
    recipe_ext.RecipeDeleteEmptyIngredients,
    recipe_ext.RecipeDeleteFavoriteRecipes,
    recipe_ext.RecipeDeleteFavorites,
    recipe_ext.RecipeDeleteFeastRecipes,
    recipe_ext.RecipeDeleteLongPrepRecipes,
    recipe_ext.RecipeDeleteMultipleRecipesComplexLogic,
    recipe_ext.RecipeDeleteNonFavorites,
    recipe_ext.RecipeDeleteQuickRecipes,
    recipe_ext.RecipeDeleteRecipesFromSource,
    recipe_ext.RecipeDeleteRecipesWithKeyword,
    recipe_ext.RecipeDeleteSoloMeals,
    recipe_ext.RecipeDeleteTitleStartsWith,
    # No complete trial:
    # recipe_ext.RecipeDeduplicateByTitle,

    # ------------------------------------------------------------- Retro Music
    retro_music_ext.RetroAddToFavorites,
    retro_music_ext.RetroClearQueue,
    retro_music_ext.RetroConsolidatePlaylists,
    retro_music_ext.RetroCreateAndRemoveSong,
    retro_music_ext.RetroCreateAndRenamePlaylist,
    retro_music_ext.RetroCreateFavoritesPlaylist,
    retro_music_ext.RetroCreatePlaylistByArtist,
    retro_music_ext.RetroCreatePlaylistExcludeSong,
    retro_music_ext.RetroCreatePlaylistExcluding,
    retro_music_ext.RetroCreatePlaylistFromArtist,
    retro_music_ext.RetroCreatePlaylistFromQueue,
    retro_music_ext.RetroCreatePlaylistLimit,
    retro_music_ext.RetroCreatePlaylistReverseAlpha,
    retro_music_ext.RetroCreatePlaylistReverseOrder,
    retro_music_ext.RetroCreatePlaylistShortSongs,
    retro_music_ext.RetroCreatePlaylistSortedAlpha,
    retro_music_ext.RetroCreatePlaylistSpecificDurationRange,
    retro_music_ext.RetroCreateTwoPlaylists,
    retro_music_ext.RetroDeletePlaylist,
    retro_music_ext.RetroDuplicatePlaylist,
    retro_music_ext.RetroMergePlaylists,
    retro_music_ext.RetroPlaySingleSong,
    retro_music_ext.RetroQueueArtist,
    retro_music_ext.RetroQueueMultipleSongs,
    retro_music_ext.RetroQueueSpecificOrder,
    retro_music_ext.RetroReorderPlaylist,
    # No complete trial:
    # retro_music_ext.RetroAddAlbumToPlaylist,
    # retro_music_ext.RetroPlaySpecificSong,
  )

  def register_task(
      self, task_registry: dict[Any, Any], task_class: type[task_eval.TaskEval]
  ) -> None:
    """Registers the task class.

    Args:
      task_registry: The registry to register the task in.
      task_class: The class to register.
    """
    task_registry[task_class.__name__] = task_class

  def __init__(self):
    # Which task set `--suite_family=android_world_extension` serves. Set
    # EXT_SUITE above to switch; exactly one is registered at a time.
    if self.EXT_SUITE == 'aw':
      tasks = self._EXT_AW_TASKS
    elif self.EXT_SUITE == 'new_app':
      tasks = self._EXT_NEW_APP_TASKS
    else:
      raise ValueError(
          f'Unsupported EXT_SUITE: {self.EXT_SUITE!r}. Expected one of'
          " 'aw', 'new_app'."
      )
    for task in tasks:
      self.register_task(self.ANDROID_EXT_TASK_REGISTRY, task)
    # Standalone OSReward tasks, registered regardless of suite.
    self.register_task(
        self.ANDROID_EXT_TASK_REGISTRY,
        settings_ext.SettingsGoogleAppNotificationsAndData,
    )

  # Add names with "." notation for autocomplete in Colab.
  names = types.SimpleNamespace(**{k: k for k in ANDROID_EXT_TASK_REGISTRY})
