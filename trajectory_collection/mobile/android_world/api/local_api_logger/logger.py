"""
Core logging module.
Provides lightweight logging of LLM API calls.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading


class APILogger:
    """Lightweight API call logger."""

    def __init__(self, log_dir: str = "api_logs"):
        """
        Initialize the logger.

        Args:
            log_dir: Log storage directory, defaults to api_logs under the current directory.
        """
        self.log_dir = Path(log_dir)
        self._lock = threading.Lock()  # thread safety

    def log_call(
        self,
        model: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        user: str = "default",
        duration_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None
    ):
        """
        Log one API call.
        Args:
            model: Model name, e.g. "claude-3-opus".
            request_data: Request data (dict).
            response_data: Response data (dict).
            user: User identifier, defaults to "default".
            duration_ms: Call duration in milliseconds.
            success: Whether the call succeeded.
            error: Error message, if any.
            metadata: Extra metadata.
        """
        timestamp = datetime.now()

        # Compute character counts
        prompt_chars = self._calculate_prompt_chars(request_data)
        completion_chars = self._calculate_completion_chars(response_data)

        # Extract token usage from the response (exact provider values only, no estimation)
        token_usage = self._extract_token_usage(response_data)
        prompt_tokens = token_usage["prompt_tokens"]
        completion_tokens = token_usage["completion_tokens"]
        total_tokens = token_usage["total_tokens"]

        # Build log entry
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "model": model,
            "user": user,
            "api_key": api_key,
            "request": request_data,
            "response": response_data,
            "prompt_chars": prompt_chars,
            "completion_chars": completion_chars,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
            "success": success,
            "error": error
        }

        if metadata:
            log_entry["metadata"] = metadata

        # Write full log
        self._write_full_log(timestamp, model, log_entry)

        # Write stats log
        self._write_stats_log(timestamp, model, user, {
            "timestamp": timestamp.isoformat(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
            "success": success
        })

    def _write_full_log(self, timestamp: datetime, model: str, log_entry: Dict[str, Any]):
        """Write the full log entry to a file."""
        # Log path: calls/{model}/{YYYY-MM}/{YYYY-MM-DD}.jsonl
        year_month = timestamp.strftime("%Y-%m")
        date = timestamp.strftime("%Y-%m-%d")

        log_path = self.log_dir / "calls" / model / year_month
        log_path.mkdir(parents=True, exist_ok=True)

        log_file = log_path / f"{date}.jsonl"

        # Thread-safe write
        with self._lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _write_stats_log(
        self,
        timestamp: datetime,
        model: str,
        user: str,
        stats_entry: Dict[str, Any]
    ):
        """Write the stats log entry."""
        # Stats log path: stats/{model}/{user}_{YYYY-MM}.jsonl
        year_month = timestamp.strftime("%Y-%m")

        stats_path = self.log_dir / "stats" / model
        stats_path.mkdir(parents=True, exist_ok=True)

        stats_file = stats_path / f"{user}_{year_month}.jsonl"

        # Thread-safe write
        with self._lock:
            with open(stats_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats_entry, ensure_ascii=False) + "\n")

    def _calculate_prompt_chars(self, request_data: Dict[str, Any]) -> int:
        """Count prompt characters in the request."""
        total_chars = 0

        # OpenAI messages format
        if "messages" in request_data:
            for message in request_data["messages"]:
                if "content" in message:
                    content = message["content"]
                    if isinstance(content, str):
                        total_chars += len(content)
                    elif isinstance(content, list):
                        # Multimodal content
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                total_chars += len(item["text"])

        # Plain prompt field
        elif "prompt" in request_data:
            total_chars = len(str(request_data["prompt"]))

        return total_chars

    def _calculate_completion_chars(self, response_data: Dict[str, Any]) -> int:
        """Count completion characters in the response."""
        total_chars = 0

        # OpenAI choices format
        if "choices" in response_data:
            for choice in response_data["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    total_chars += len(str(choice["message"]["content"]))
                elif "text" in choice:
                    total_chars += len(str(choice["text"]))

        # Direct content field
        elif "content" in response_data:
            total_chars = len(str(response_data["content"]))

        return total_chars

    def _extract_token_usage(self, response_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract token usage from the response.

        Uses the exact token counts returned by the provider; returns 0 if absent (no estimation).
        """
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        if "usage" in response_data and response_data["usage"] is not None:
            usage = response_data["usage"]
            token_usage["prompt_tokens"] = usage.get("prompt_tokens", 0)
            token_usage["completion_tokens"] = usage.get("completion_tokens", 0)
            token_usage["total_tokens"] = usage.get("total_tokens", 0)

        return token_usage


# Global default instance
_default_logger = APILogger()

def log_call(*args, **kwargs):
    """Log a call using the default logger."""
    return _default_logger.log_call(*args, **kwargs)


def set_log_dir(log_dir: str):
    """Set the global log directory."""
    global _default_logger
    _default_logger = APILogger(log_dir)

# log_completion = log_call