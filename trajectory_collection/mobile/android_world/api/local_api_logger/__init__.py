"""
Local API Logger - lightweight LLM API call logging tool.

A minimal Python library for logging LLM API calls locally, including:
- Request and response data
- Token usage statistics
- Call duration and success rate
- Statistics grouped by model and user

Features:
- Zero configuration, works out of the box
- No database required; stores data in JSONL format
- Thread-safe
- Supports streaming and non-streaming responses
- Provides decorators and wrappers for easy integration

Basic usage:
    ```python
    from local_api_logger import log_completion, print_stats_summary

    # Log one call
    log_completion(
        model="claude-3-opus",
        request_data={"messages": [...]},
        response_data=response,  # full provider response (includes usage)
        user="john"
    )

    # View statistics
    print_stats_summary()
    ```
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

# Export core classes
from .logger import APILogger, log_call, set_log_dir
from .tracker import APITracker, track_request, log_completion, wrap_requests_call
from .viewer import LogViewer, get_stats_summary, print_stats_summary, print_recent_calls, export_to_csv
from .utils import (
    estimate_tokens,
    format_token_count,
    calculate_cost,
    truncate_string,
    extract_model_name,
    safe_get
)

# Public API
__all__ = [
    # Version info
    "__version__",

    # Core classes
    "APILogger",
    "APITracker",
    "LogViewer",

    # Logger functions
    "log_call",
    "set_log_dir",

    # Tracker functions
    "track_request",
    "log_completion",
    "wrap_requests_call",

    # Viewer functions
    "get_stats_summary",
    "print_stats_summary",
    "print_recent_calls",
    "export_to_csv",

    # Utility functions
    "estimate_tokens",
    "format_token_count",
    "calculate_cost",
    "truncate_string",
    "extract_model_name",
    "safe_get",
]
