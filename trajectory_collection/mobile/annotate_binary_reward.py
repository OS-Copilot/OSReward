#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Any, Dict, Optional, Tuple, List

TASK_REWARD_MAP = {
    "ClockTimerEntryComplex": 0,
    "ClockTimerEntryFifteenMinutes": 1,
    "ClockTimerEntryFiveMinutes": 1,
    "ClockTimerEntryFortyFiveSeconds": 1,
    "ClockTimerEntryMaxTime": 0,
    "ClockTimerEntryOneHour": 1,
    "ClockTimerEntryOneHourThirtyMinutes": 0,
    "ClockTimerEntryPreciseCooking": 0,
    "ClockTimerEntryThirtyMinutes": 0,

    "ExpenseAddHighValueSingle": 1,
    "ExpenseAddMonthlySubscriptions": 1,
    "ExpenseAddMultipleFoodCategory": 1,
    "ExpenseAddWithSpecificNote": 1,
    "ExpenseDeleteHighValueExpenses": 1,

    "MarkorArchiveNote": 0,
    "MarkorConvertTxtToMd": 0,
    "MarkorCreateDailyLog": 0,
    "MarkorCreateShoppingList": 0,
    "MarkorDeleteOldestNote": 0,
    "MarkorDeleteSpecificFolder": 0,
    "MarkorDuplicateNote": 0,
    "MarkorRenameNote": 0,

    "RecipeAddFavoriteRecipe": 1,
    "RecipeAddHighProteinRecipe": 1,
    "RecipeAddPartyRecipe": 1,
    "RecipeAddQuickMeal": 1,
    "RecipeAddSourceRecipe": 1,
    "RecipeDeleteFavoriteRecipes": 1,
    "RecipeDeleteLongPrepRecipes": 1,
    "RecipeDeleteRecipesFromSource": 1,
    "RecipeDeleteRecipesWithKeyword": 1,
    "RecipeDeleteSoloMeals": 1,

    "RetroCreateFavoritesPlaylist": 0,
    "RetroCreatePlaylistExcludeSong": 0,
    "RetroCreatePlaylistFromArtist": 0,
    "RetroCreatePlaylistReverseOrder": 1,
    "RetroCreatePlaylistShortSongs": 0,
    "RetroCreateTwoPlaylists": 0,
    "RetroDuplicatePlaylist": 1,
    "RetroPlaySingleSong": 1,
    "RetroQueueArtist": 1,
    "RetroQueueSpecificOrder": 1,

    "SimpleCalendarAddEveningEvent": 1,
    "SimpleCalendarAddLongEvent": 1,
    "SimpleCalendarAddMorningEvent": 1,
    "SimpleCalendarAddOneEventNextWeek": 1,
    "SimpleCalendarAddOneEventThisWeekend": 1,
    "SimpleCalendarAddOneEventToday": 1,
    "SimpleCalendarDeleteEventsNextWeek": 0,
    "SimpleCalendarDeleteEventsThisWeekend": 1,
    "SimpleCalendarDeleteEventsTomorrow": 1,
}

TASK_REWARD_MAP_UPDATE_2 = {
    "ExpenseAddBackdated": 1,
    "ExpenseAddBudgetCorrection": 1,
    "ExpenseAddBusinessTrip": 1,
    "ExpenseAddDetailed": 1,
    "ExpenseAddMonthlyBills": 1,
    "ExpenseAddPartyExpenses": 1,
    "ExpenseAddSpecificDate": 1,
    "ExpenseCleanupMisc": 0,
    "ExpenseDeduplicateStrict": 1,
    "ExpenseDeleteAllExceptFood": 1,
    "ExpenseDeleteAllIncome": 1,
    "ExpenseDeleteByDate": 1,
    "ExpenseDeleteByNameSubstring": 1,
    "ExpenseDeleteEntertainmentAndSocial": 0,
    "ExpenseDeleteFood": 0,
    "ExpenseDeleteHousing": 0,
    "ExpenseDeleteLowValue": 1,
    "ExpenseDeletePaidByCard": 1,
    "ExpenseDeleteUrgent": 0,

    "MarkorAppendDateToName": 0,
    "MarkorBatchRenamePrefix": 0,
    "MarkorConvertListToTasks": 1,
    "MarkorCreateIndex": 0,
    "MarkorCreateNestedFolder": 1,
    "MarkorCreateSummaryNote": 0,
    "MarkorDailyLogCreation": 1,
    "MarkorDeleteEmptyNotes": 1,
    "MarkorDeleteNotesContainingText": 0,
    "MarkorMoveDoneTasks": 0,
    "MarkorReplaceTextBatch": 0,
    "MarkorSplitNote": 0,
    "MarkorSwapContent": 0,
    "MarkorTemplateNote": 1,

    "RecipeAddDetailed": 1,
    "RecipeAddLargeBatch": 1,
    "RecipeAddMarkorByIngredient": 1,
    "RecipeAddMarkorFavorites": 1,
    "RecipeAddMarkorHealthy": 1,
    "RecipeAddMultipleRecipesFromMarkorQuick": 1,
    "RecipeDeleteByDescriptionKeyword": 1,
    "RecipeDeleteByPrepTime": 1,
    "RecipeDeleteByServings": 1,
    "RecipeDeleteBySource": 0,
    "RecipeDeleteByTitleSubstring": 1,
    "RecipeDeleteEmptyIngredients": 1,
    "RecipeDeleteFavorites": 1,
    "RecipeDeleteFeastRecipes": 1,
    "RecipeDeleteMultipleRecipesComplexLogic": 1,
    "RecipeDeleteNonFavorites": 1,
    "RecipeDeleteQuickRecipes": 1,
    "RecipeDeleteTitleStartsWith": 1,

    "RetroAddToFavorites": 1,
    "RetroClearQueue": 1,
    "RetroConsolidatePlaylists": 0,
    "RetroCreateAndRemoveSong": 1,
    "RetroCreateAndRenamePlaylist": 0,
    "RetroCreatePlaylistByArtist": 1,
    "RetroCreatePlaylistExcluding": 0,
    "RetroCreatePlaylistFromQueue": 0,
    "RetroCreatePlaylistLimit": 0,
    "RetroCreatePlaylistReverseAlpha": 0,
    "RetroCreatePlaylistSortedAlpha": 0,
    "RetroCreatePlaylistSpecificDurationRange": 1,
    "RetroCreateTwoPlaylists": 0,
    "RetroDeletePlaylist": 0,
    "RetroDuplicatePlaylist": 0,
    "RetroMergePlaylists": 0,
    "RetroQueueMultipleSongs": 1,
    "RetroReorderPlaylist": 0,

    "SimpleCalendarAddAllDayEvent": 1,
    "SimpleCalendarAddEventDurationHours": 1,
    "SimpleCalendarAddEventWithLocation": 1,
    "SimpleCalendarAddEventWithLongDescription": 1,
    "SimpleCalendarAddEventWithStartEnd": 1,
    "SimpleCalendarAddOverlappingEvent": 1,
    "SimpleCalendarAddThreeEventsDifferentDays": 1,
    "SimpleCalendarAddTwoEventsSameDay": 1,
    "SimpleCalendarAddWeeklyMeetingWithEnd": 1,
    "SimpleCalendarDeleteAllEventsInWeek": 1,
    "SimpleCalendarDeleteEventByDescription": 1,
    "SimpleCalendarDeleteEventByLocation": 1,
    "SimpleCalendarDeleteEventByTitle": 1,
}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def iter_json_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                yield os.path.join(dirpath, fn)

def extract_task_name(ep: Dict[str, Any]) -> Optional[str]:
    """
    Extract task name from:
      1) ep["task_id"] (possibly ClockTimerEntryComplex_0)
      2) ep["original"]["task_template"]
      3) ep["task_template"]

    Normalize by taking prefix before first '_' if present.
    """
    raw = ep.get("task_id") or ""
    if not raw:
        raw = (ep.get("original") or {}).get("task_template") or ep.get("task_template") or ""

    if not isinstance(raw, str) or not raw:
        return None

    return raw.split("_", 1)[0]

def ensure_orm_label(ep: Dict[str, Any]) -> None:
    if "orm_label" not in ep or not isinstance(ep["orm_label"], dict):
        ep["orm_label"] = {"score": None, "binary_reward": None, "rationale": ""}
    ep["orm_label"].setdefault("score", None)
    ep["orm_label"].setdefault("binary_reward", None)
    ep["orm_label"].setdefault("rationale", "")

def apply_reward(ep: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[int], Optional[int]]:
    """
    Returns a status string + task_name + old_val + new_val
    status in {"UPDATED","UNCHANGED","SKIPPED"}
    """
    task_name = extract_task_name(ep)
    if not task_name or task_name not in TASK_REWARD_MAP_UPDATE_2:
        return "SKIPPED", task_name, None, None

    ensure_orm_label(ep)
    new_val = int(TASK_REWARD_MAP_UPDATE_2[task_name])
    old_val = ep["orm_label"].get("binary_reward", None)

    if old_val != new_val:
        ep["orm_label"]["binary_reward"] = new_val
        return "UPDATED", task_name, old_val, new_val

    return "UNCHANGED", task_name, old_val, new_val

def log(line: str, report_lines: List[str], report_path: Optional[str]):
    print(line)
    if report_path is not None:
        report_lines.append(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory of reformatted trajectory JSON files.")
    ap.add_argument("--dry_run", action="store_true", help="Do not write changes; only report.")
    ap.add_argument("--report", default=None, help="Optional path to save a full log report (txt).")
    ap.add_argument("--show_unchanged", action="store_true", help="Also print UNCHANGED entries (can be verbose).")
    args = ap.parse_args()

    report_lines: List[str] = []

    updated_files = 0
    unchanged_files = 0
    skipped_files = 0
    failed_files = 0

    updated_eps = 0
    unchanged_eps = 0
    skipped_eps = 0

    for path in iter_json_files(args.root):
        try:
            data = load_json(path)

            file_any_known = False
            file_any_updated = False
            file_any_error = False

            # Track per-file episode outcomes
            per_file_msgs = []

            def process_episode(ep: Dict[str, Any]) -> None:
                nonlocal file_any_known, file_any_updated, updated_eps, unchanged_eps, skipped_eps

                status, task_name, old_val, new_val = apply_reward(ep)

                if status == "SKIPPED":
                    skipped_eps += 1
                    per_file_msgs.append(f"  ⚠️  SKIPPED_EP task={task_name} (not in table)")
                    return

                file_any_known = True
                if status == "UPDATED":
                    updated_eps += 1
                    file_any_updated = True
                    per_file_msgs.append(f"  ✅ UPDATED_EP task={task_name} binary_reward {old_val} -> {new_val}")
                else:
                    unchanged_eps += 1
                    per_file_msgs.append(f"  ➖ UNCHANGED_EP task={task_name} binary_reward={new_val}")

            if isinstance(data, dict) and "trajectory" in data:
                process_episode(data)

            elif isinstance(data, list):
                # multi-episode file
                for ep in data:
                    if isinstance(ep, dict) and "trajectory" in ep:
                        process_episode(ep)

            else:
                # Not a reformatted episode file; ignore silently
                continue

            # File-level decision
            if not file_any_known:
                skipped_files += 1
                log(f"⚠️  SKIPPED_FILE (no known tasks in file): {path}", report_lines, args.report)
                # show the first few episode messages for debugging
                for m in per_file_msgs[:10]:
                    log(m, report_lines, args.report)
                continue

            if file_any_updated:
                updated_files += 1
                log(f"✅ UPDATED_FILE: {path}", report_lines, args.report)
                for m in per_file_msgs[:10]:
                    log(m, report_lines, args.report)

                if not args.dry_run:
                    try:
                        save_json(path, data)
                    except Exception as e:
                        failed_files += 1
                        log(f"❌ ERROR_WRITE: {path} | {repr(e)}", report_lines, args.report)
                        continue
            else:
                unchanged_files += 1
                if args.show_unchanged:
                    log(f"➖ UNCHANGED_FILE: {path}", report_lines, args.report)
                    for m in per_file_msgs[:10]:
                        log(m, report_lines, args.report)

        except Exception as e:
            failed_files += 1
            log(f"❌ ERROR_READ_OR_PARSE: {path} | {repr(e)}", report_lines, args.report)

    # Write report if requested
    if args.report is not None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
            with open(args.report, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines) + "\n")
            print(f"\n[REPORT] Saved log to: {args.report}")
        except Exception as e:
            print(f"\n[WARN] Failed to write report: {repr(e)}")

    print("\n===== Summary =====")
    print(f"Dry run              : {args.dry_run}")
    print(f"Files updated        : {updated_files}")
    print(f"Files unchanged      : {unchanged_files}")
    print(f"Files skipped        : {skipped_files}")
    print(f"Files failed         : {failed_files}")
    print("---")
    print(f"Episodes updated     : {updated_eps}")
    print(f"Episodes unchanged   : {unchanged_eps}")
    print(f"Episodes skipped     : {skipped_eps}")

if __name__ == "__main__":
    main()
