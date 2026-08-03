"""
Log viewing and analysis tools.
Provides statistics, queries, and export.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class LogViewer:
    """Log viewer."""

    def __init__(self, log_dir: str = "api_logs"):
        """
        Initialize the log viewer.

        Args:
            log_dir: Log directory.
        """
        self.log_dir = Path(log_dir)

    def get_stats_summary(self, model: Optional[str] = None, month: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a statistics summary.

        Args:
            model: Model name (optional; aggregates all models if omitted).
            month: Month in YYYY-MM format (optional).

        Returns:
            Statistics summary dict.
        """
        stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "zero_output_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "by_model": defaultdict(lambda: {
                "calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "zero_output_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }),
            "by_user": defaultdict(lambda: {
                "calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "zero_output_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
        }

        stats_path = self.log_dir / "stats"
        if not stats_path.exists():
            return stats

        # Iterate over all stats files
        for model_dir in stats_path.iterdir():
            if not model_dir.is_dir():
                continue

            current_model = model_dir.name

            # Skip other models if a model is specified
            if model and current_model != model:
                continue

            for stats_file in model_dir.glob("*.jsonl"):
                # Filter by month if specified
                if month and month not in stats_file.stem:
                    continue

                # Extract user name from the file name
                user = stats_file.stem.rsplit('_', 2)[0]

                with open(stats_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())

                            prompt_tokens = entry.get("prompt_tokens", 0)
                            completion_tokens = entry.get("completion_tokens", 0)
                            total_tokens = entry.get("total_tokens", 0)
                            success = entry.get("success", True)

                            # Count all calls, including failed ones
                            stats["total_calls"] += 1
                            stats["by_model"][current_model]["calls"] += 1
                            stats["by_user"][user]["calls"] += 1

                            # Count successes/failures
                            if success:
                                stats["successful_calls"] += 1
                                stats["by_model"][current_model]["successful_calls"] += 1
                                stats["by_user"][user]["successful_calls"] += 1
                            else:
                                stats["failed_calls"] += 1
                                stats["by_model"][current_model]["failed_calls"] += 1
                                stats["by_user"][user]["failed_calls"] += 1

                            # Count calls with zero output tokens
                            if completion_tokens == 0:
                                stats["zero_output_calls"] += 1
                                stats["by_model"][current_model]["zero_output_calls"] += 1
                                stats["by_user"][user]["zero_output_calls"] += 1

                            # Accumulate token counts
                            stats["total_prompt_tokens"] += prompt_tokens
                            stats["total_completion_tokens"] += completion_tokens
                            stats["total_tokens"] += total_tokens

                            stats["by_model"][current_model]["prompt_tokens"] += prompt_tokens
                            stats["by_model"][current_model]["completion_tokens"] += completion_tokens
                            stats["by_model"][current_model]["total_tokens"] += total_tokens

                            stats["by_user"][user]["prompt_tokens"] += prompt_tokens
                            stats["by_user"][user]["completion_tokens"] += completion_tokens
                            stats["by_user"][user]["total_tokens"] += total_tokens

                        except Exception:
                            continue

        # Convert defaultdicts to plain dicts
        stats["by_model"] = dict(stats["by_model"])
        stats["by_user"] = dict(stats["by_user"])

        return stats

    def get_recent_calls(self, model: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent API call records.

        Args:
            model: Model name (optional).
            limit: Maximum number of records to return.

        Returns:
            List of call records.
        """
        calls = []
        calls_path = self.log_dir / "calls"

        if not calls_path.exists():
            return calls

        # Collect all log files
        log_files = []
        for model_dir in calls_path.iterdir():
            if not model_dir.is_dir():
                continue

            # Skip other models if a model is specified
            if model and model_dir.name != model:
                continue

            for month_dir in model_dir.iterdir():
                if not month_dir.is_dir():
                    continue

                for log_file in month_dir.glob("*.jsonl"):
                    log_files.append(log_file)

        # Sort by modification time, newest first
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Read the most recent records
        for log_file in log_files:
            if len(calls) >= limit:
                break

            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                # Read from the end of the file
                for line in reversed(lines):
                    if len(calls) >= limit:
                        break

                    try:
                        entry = json.loads(line.strip())
                        calls.append(entry)
                    except Exception:
                        continue

        return calls[:limit]

    def print_stats_summary(self, model: Optional[str] = None, month: Optional[str] = None):
        """
        Print a formatted statistics summary.

        Args:
            model: Model name (optional).
            month: Month (optional).
        """
        stats = self.get_stats_summary(model, month)

        print("=" * 80)
        print("API Call Statistics Summary")
        if model:
            print(f"Model: {model}")
        if month:
            print(f"Month: {month}")
        print("=" * 80)

        # Compute success rate
        success_rate = 0
        if stats['total_calls'] > 0:
            success_rate = (stats['successful_calls'] / stats['total_calls']) * 100

        print(f"\nTotal calls: {stats['total_calls']:,}")
        print(f"  ✓ Successful: {stats['successful_calls']:,}")
        print(f"  ✗ Failed: {stats['failed_calls']:,}")
        print(f"  ⚠ Zero output tokens: {stats['zero_output_calls']:,}")
        print(f"  Success rate: {success_rate:.1f}%")

        print(f"\nTotal input tokens: {stats['total_prompt_tokens']:,}")
        print(f"Total output tokens: {stats['total_completion_tokens']:,}")
        print(f"Total tokens: {stats['total_tokens']:,}")

        if stats["by_model"]:
            print("\nBy model:")
            print("-" * 80)
            for model_name, model_stats in stats["by_model"].items():
                model_success_rate = 0
                if model_stats['calls'] > 0:
                    model_success_rate = (model_stats['successful_calls'] / model_stats['calls']) * 100

                print(f"\n{model_name}:")
                print(f"  Calls: {model_stats['calls']:,} (successful: {model_stats['successful_calls']:,}, failed: {model_stats['failed_calls']:,})")
                print(f"  Success rate: {model_success_rate:.1f}%")
                print(f"  Zero output tokens: {model_stats['zero_output_calls']:,}")
                print(f"  Input tokens: {model_stats['prompt_tokens']:,}")
                print(f"  Output tokens: {model_stats['completion_tokens']:,}")
                print(f"  Total tokens: {model_stats['total_tokens']:,}")

        if stats["by_user"]:
            print("\nBy user:")
            print("-" * 80)
            for user_name, user_stats in stats["by_user"].items():
                user_success_rate = 0
                if user_stats['calls'] > 0:
                    user_success_rate = (user_stats['successful_calls'] / user_stats['calls']) * 100

                print(f"\n{user_name}:")
                print(f"  Calls: {user_stats['calls']:,} (successful: {user_stats['successful_calls']:,}, failed: {user_stats['failed_calls']:,})")
                print(f"  Success rate: {user_success_rate:.1f}%")
                print(f"  Zero output tokens: {user_stats['zero_output_calls']:,}")
                print(f"  Input tokens: {user_stats['prompt_tokens']:,}")
                print(f"  Output tokens: {user_stats['completion_tokens']:,}")
                print(f"  Total tokens: {user_stats['total_tokens']:,}")

        print("\n" + "=" * 80)

    def print_recent_calls(self, model: Optional[str] = None, limit: int = 5):
        """
        Print recent call records.

        Args:
            model: Model name (optional).
            limit: Number of records to display.
        """
        calls = self.get_recent_calls(model, limit)

        print("=" * 80)
        print(f"Last {limit} API calls")
        if model:
            print(f"Model: {model}")
        print("=" * 80)

        for i, call in enumerate(calls, 1):
            print(f"\n--- Call #{i} ---")
            print(f"Time: {call.get('timestamp', 'N/A')}")
            print(f"Model: {call.get('model', 'N/A')}")
            print(f"User: {call.get('user', 'N/A')}")
            print(f"Success: {'yes' if call.get('success', False) else 'no'}")

            if call.get('error'):
                print(f"Error: {call['error']}")

            print(f"Input tokens: {call.get('prompt_tokens', 0):,}")
            print(f"Output tokens: {call.get('completion_tokens', 0):,}")
            print(f"Total tokens: {call.get('total_tokens', 0):,}")

            if call.get('duration_ms') is not None:
                print(f"Duration: {call['duration_ms']:.2f} ms")

        print("\n" + "=" * 80)

    def export_to_csv(
        self,
        output_file: str,
        model: Optional[str] = None,
        month: Optional[str] = None
    ):
        """
        Export statistics to CSV.

        Args:
            output_file: Output file path.
            model: Model name (optional).
            month: Month (optional).
        """
        import csv

        stats_path = self.log_dir / "stats"
        if not stats_path.exists():
            print("No statistics data found")
            return

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "timestamp", "model", "user",
                "prompt_tokens", "completion_tokens", "total_tokens", "duration_ms", "success"
            ])

            # Iterate over stats files
            for model_dir in stats_path.iterdir():
                if not model_dir.is_dir():
                    continue

                current_model = model_dir.name
                if model and current_model != model:
                    continue

                for stats_file in model_dir.glob("*.jsonl"):
                    if month and month not in stats_file.stem:
                        continue

                    user = stats_file.stem.rsplit('_', 2)[0]

                    with open(stats_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                writer.writerow([
                                    entry.get("timestamp", ""),
                                    current_model,
                                    user,
                                    entry.get("prompt_tokens", 0),
                                    entry.get("completion_tokens", 0),
                                    entry.get("total_tokens", 0),
                                    entry.get("duration_ms", ""),
                                    "yes" if entry.get("success", True) else "no"
                                ])
                            except Exception:
                                continue

        print(f"Data exported to: {output_file}")


# Global default viewer
_default_viewer = LogViewer()


def get_stats_summary(model: Optional[str] = None, month: Optional[str] = None) -> Dict[str, Any]:
    """Get a statistics summary using the default viewer."""
    return _default_viewer.get_stats_summary(model, month)


def print_stats_summary(model: Optional[str] = None, month: Optional[str] = None):
    """Print a statistics summary using the default viewer."""
    _default_viewer.print_stats_summary(model, month)


def print_recent_calls(model: Optional[str] = None, limit: int = 5):
    """Print recent calls using the default viewer."""
    _default_viewer.print_recent_calls(model, limit)


def export_to_csv(output_file: str, model: Optional[str] = None, month: Optional[str] = None):
    """Export data using the default viewer."""
    _default_viewer.export_to_csv(output_file, model, month)
