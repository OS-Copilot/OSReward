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

"""Registers the task classes.

Only the OSReward extension tasks (task_evals/single/extensions/) are
registered; the stock AndroidWorld tasks stay in the tree solely as base
classes for them.
"""

import types
from typing import Any, Final

from android_world.task_evals import task_eval
from android_world.task_evals.single.extensions import extensions_calendar as calendar_ext
from android_world.task_evals.single.extensions import extensions_calendar2 as calendar_ext2
from android_world.task_evals.single.extensions import extensions_clock as clock_ext
from android_world.task_evals.single.extensions import extensions_expense as expense_ext
from android_world.task_evals.single.extensions import extensions_expense2 as expense_ext2
from android_world.task_evals.single.extensions import extensions_markor as markor_ext
from android_world.task_evals.single.extensions import extensions_markor2 as markor_ext2
from android_world.task_evals.single.extensions import extensions_recipe as recipe_ext
from android_world.task_evals.single.extensions import extensions_recipe2 as recipe_ext2
from android_world.task_evals.single.extensions import extensions_retro_music as retro_music_ext
from android_world.task_evals.single.extensions import extensions_retro_music2 as retro_music_ext2
from android_world.task_evals.single.extensions import extensions_settings as settings_ext


def get_families() -> list[str]:
  return [
      TaskRegistry.ANDROID_WORLD_FAMILY,
      TaskRegistry.ANDROID_FAMILY,
  ]


class TaskRegistry:
  """Registry of tasks."""

  # The AndroidWorld family.
  ANDROID_WORLD_FAMILY: Final[str] = 'android_world'  # Entire suite.
  ANDROID_FAMILY: Final[str] = 'android'  # Subset.

  # Task registries; they contain a mapping from each task name to its class,
  # to construct instances of a task.
  ANDROID_TASK_REGISTRY = {}

  def get_registry(self, family: str) -> Any:
    """Gets the task registry for the given family.

    Args:
      family: The family.

    Returns:
      Task registry.

    Raises:
      ValueError: If provided family doesn't exist.
    """
    if family in (self.ANDROID_WORLD_FAMILY, self.ANDROID_FAMILY):
      return self.ANDROID_TASK_REGISTRY
    else:
      raise ValueError(f'Unsupported family: {family}')

  _EXTENSION_TASKS = (
      # keep-sorted start
      clock_ext.ClockTimerEntryComplex,
      clock_ext.ClockTimerEntryFifteenMinutes,
      clock_ext.ClockTimerEntryFiveMinutes,
      clock_ext.ClockTimerEntryFortyFiveSeconds,
      clock_ext.ClockTimerEntryLongDuration,
      clock_ext.ClockTimerEntryMaxTime,
      clock_ext.ClockTimerEntryOneHour,
      clock_ext.ClockTimerEntryOneHourThirtyMinutes,
      clock_ext.ClockTimerEntryPreciseCooking,
      clock_ext.ClockTimerEntryThirtyMinutes,
      calendar_ext.SimpleCalendarAddEveningEvent,
      calendar_ext.SimpleCalendarAddLongEvent,
      calendar_ext.SimpleCalendarAddMorningEvent,
      # calendar_ext.SimpleCalendarAddOneEventInOneMonth: generate_random_params crashes: day+30 overflows the month.
      calendar_ext.SimpleCalendarAddOneEventNextWeek,
      calendar_ext.SimpleCalendarAddOneEventThisWeekend,
      calendar_ext.SimpleCalendarAddOneEventToday,
      calendar_ext.SimpleCalendarDeleteEventsNextWeek,
      calendar_ext.SimpleCalendarDeleteEventsThisWeekend,
      calendar_ext.SimpleCalendarDeleteEventsTomorrow,
      calendar_ext2.SimpleCalendarAddAllDayEvent,
      calendar_ext2.SimpleCalendarAddEventDurationHours,
      calendar_ext2.SimpleCalendarAddEventWithLocation,
      calendar_ext2.SimpleCalendarAddEventWithLongDescription,
      calendar_ext2.SimpleCalendarAddEventWithStartEnd,
      calendar_ext2.SimpleCalendarAddOverlappingEvent,
      calendar_ext2.SimpleCalendarAddThreeEventsDifferentDays,
      calendar_ext2.SimpleCalendarAddTwoEventsSameDay,
      calendar_ext2.SimpleCalendarAddWeeklyMeetingWithEnd,
      calendar_ext2.SimpleCalendarClearMonth,
      calendar_ext2.SimpleCalendarDeleteAllEventsInWeek,
      calendar_ext2.SimpleCalendarDeleteAllFutureEvents,
      calendar_ext2.SimpleCalendarDeleteEventByDescription,
      calendar_ext2.SimpleCalendarDeleteEventByLocation,
      calendar_ext2.SimpleCalendarDeleteEventByTitle,
      # expense_ext.ExpenseAddEmergencyRepair: excluded in original collection.
      expense_ext.ExpenseAddHighValueSingle,
      expense_ext.ExpenseAddMonthlySubscriptions,
      expense_ext.ExpenseAddMultipleFoodCategory,
      expense_ext.ExpenseAddVacationExpenses,
      expense_ext.ExpenseAddWithSpecificNote,
      expense_ext.ExpenseDeleteHighValueExpenses,
      # expense_ext.ExpenseDeleteHousingCategory: excluded in original collection.
      expense_ext.ExpenseDeleteSmallExpenses,
      expense_ext.ExpenseDeleteSpecificNote,
      expense_ext2.ExpenseAddBackdated,
      expense_ext2.ExpenseAddBudgetCorrection,
      expense_ext2.ExpenseAddBusinessTrip,
      expense_ext2.ExpenseAddDetailed,
      expense_ext2.ExpenseAddMonthlyBills,
      expense_ext2.ExpenseAddPartyExpenses,
      expense_ext2.ExpenseAddSpecificDate,
      expense_ext2.ExpenseCleanupMisc,
      expense_ext2.ExpenseDeduplicateStrict,
      expense_ext2.ExpenseDeleteAllExceptFood,
      expense_ext2.ExpenseDeleteAllIncome,
      expense_ext2.ExpenseDeleteByDate,
      expense_ext2.ExpenseDeleteByNameSubstring,
      expense_ext2.ExpenseDeleteEntertainmentAndSocial,
      expense_ext2.ExpenseDeleteFood,
      expense_ext2.ExpenseDeleteHousing,
      expense_ext2.ExpenseDeleteLowValue,
      expense_ext2.ExpenseDeletePaidByCard,
      expense_ext2.ExpenseDeleteUrgent,
      markor_ext.MarkorAppendToNote,
      markor_ext.MarkorArchiveNote,
      markor_ext.MarkorClearNote,
      markor_ext.MarkorConvertTxtToMd,
      markor_ext.MarkorCreateDailyLog,
      markor_ext.MarkorCreateShoppingList,
      markor_ext.MarkorDeleteOldestNote,
      markor_ext.MarkorDeleteSpecificFolder,
      markor_ext.MarkorDuplicateNote,
      markor_ext.MarkorRenameNote,
      markor_ext2.MarkorAppendDateToName,
      markor_ext2.MarkorArchiveByExtension,
      markor_ext2.MarkorBackupNote,
      markor_ext2.MarkorBatchRenamePrefix,
      markor_ext2.MarkorConvertListToTasks,
      markor_ext2.MarkorCopyMultipleFiles,
      markor_ext2.MarkorCreateIndex,
      markor_ext2.MarkorCreateNestedFolder,
      markor_ext2.MarkorCreateSummaryNote,
      markor_ext2.MarkorDailyLogCreation,
      markor_ext2.MarkorDeleteEmptyNotes,
      markor_ext2.MarkorDeleteNotesContainingText,
      markor_ext2.MarkorMoveDoneTasks,
      markor_ext2.MarkorOrganizeByKeyword,
      markor_ext2.MarkorReplaceTextBatch,
      markor_ext2.MarkorRestoreFromTrash,
      markor_ext2.MarkorSeparateExtensionsTwoFolders,
      markor_ext2.MarkorSplitNote,
      markor_ext2.MarkorSwapContent,
      markor_ext2.MarkorTemplateNote,
      recipe_ext.RecipeAddFavoriteRecipe,
      recipe_ext.RecipeAddHighProteinRecipe,
      recipe_ext.RecipeAddPartyRecipe,
      recipe_ext.RecipeAddQuickMeal,
      recipe_ext.RecipeAddSourceRecipe,
      recipe_ext.RecipeDeleteFavoriteRecipes,
      recipe_ext.RecipeDeleteLongPrepRecipes,
      recipe_ext.RecipeDeleteRecipesFromSource,
      recipe_ext.RecipeDeleteRecipesWithKeyword,
      recipe_ext.RecipeDeleteSoloMeals,
      recipe_ext2.RecipeAddDetailed,
      recipe_ext2.RecipeAddLargeBatch,
      recipe_ext2.RecipeAddMarkorByIngredient,
      recipe_ext2.RecipeAddMarkorFavorites,
      recipe_ext2.RecipeAddMarkorHealthy,
      recipe_ext2.RecipeAddMultipleRecipesFromMarkorQuick,
      recipe_ext2.RecipeDeduplicateByTitle,
      recipe_ext2.RecipeDeleteByDescriptionKeyword,
      recipe_ext2.RecipeDeleteByPrepTime,
      recipe_ext2.RecipeDeleteByServings,
      recipe_ext2.RecipeDeleteBySource,
      recipe_ext2.RecipeDeleteByTitleSubstring,
      recipe_ext2.RecipeDeleteEmptyIngredients,
      recipe_ext2.RecipeDeleteFavorites,
      recipe_ext2.RecipeDeleteFeastRecipes,
      recipe_ext2.RecipeDeleteMultipleRecipesComplexLogic,
      recipe_ext2.RecipeDeleteNonFavorites,
      recipe_ext2.RecipeDeleteQuickRecipes,
      recipe_ext2.RecipeDeleteTitleStartsWith,
      retro_music_ext.RetroCreateFavoritesPlaylist,
      retro_music_ext.RetroCreatePlaylistExcludeSong,
      retro_music_ext.RetroCreatePlaylistFromArtist,
      retro_music_ext.RetroCreatePlaylistReverseOrder,
      retro_music_ext.RetroCreatePlaylistShortSongs,
      # retro_music_ext.RetroCreateTwoPlaylists superseded by retro_music_ext2.RetroCreateTwoPlaylists.
      # retro_music_ext.RetroDuplicatePlaylist superseded by retro_music_ext2.RetroDuplicatePlaylist.
      retro_music_ext.RetroPlaySingleSong,
      retro_music_ext.RetroQueueArtist,
      retro_music_ext.RetroQueueSpecificOrder,
      retro_music_ext2.RetroAddAlbumToPlaylist,
      retro_music_ext2.RetroAddToFavorites,
      retro_music_ext2.RetroClearQueue,
      retro_music_ext2.RetroConsolidatePlaylists,
      retro_music_ext2.RetroCreateAndRemoveSong,
      retro_music_ext2.RetroCreateAndRenamePlaylist,
      retro_music_ext2.RetroCreatePlaylistByArtist,
      retro_music_ext2.RetroCreatePlaylistExcluding,
      retro_music_ext2.RetroCreatePlaylistFromQueue,
      retro_music_ext2.RetroCreatePlaylistLimit,
      retro_music_ext2.RetroCreatePlaylistReverseAlpha,
      retro_music_ext2.RetroCreatePlaylistSortedAlpha,
      retro_music_ext2.RetroCreatePlaylistSpecificDurationRange,
      retro_music_ext2.RetroCreateTwoPlaylists,
      retro_music_ext2.RetroDeletePlaylist,
      retro_music_ext2.RetroDuplicatePlaylist,
      retro_music_ext2.RetroMergePlaylists,
      retro_music_ext2.RetroPlaySpecificSong,
      retro_music_ext2.RetroQueueMultipleSongs,
      retro_music_ext2.RetroReorderPlaylist,
      settings_ext.SettingsGoogleAppNotificationsAndData,
      # keep-sorted end
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
    for task in self._EXTENSION_TASKS:
      self.register_task(self.ANDROID_TASK_REGISTRY, task)

  # Add names with "." notation for autocomplete in Colab.
  names = types.SimpleNamespace(**{k: k for k in ANDROID_TASK_REGISTRY})
