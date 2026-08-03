"""
Utility functions.
Helpers for token estimation and data processing.
"""


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.

    Heuristic:
    - Chinese characters: ~2 chars/token
    - Other characters: ~4 chars/token
    - Mixed text: weighted average

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    # Count Chinese characters
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars

    # Chinese ~2 chars/token, other ~4 chars/token
    estimated_tokens = (chinese_chars / 2) + (other_chars / 4)

    return int(estimated_tokens)


def format_token_count(count: int) -> str:
    """
    Format a token count for display.

    Args:
        count: Token count.

    Returns:
        Formatted string, e.g. "1.2K" or "1.5M".
    """
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float
) -> float:
    """
    Calculate the cost of a call.

    Args:
        prompt_tokens: Input token count.
        completion_tokens: Output token count.
        input_price_per_million: Input price per million tokens.
        output_price_per_million: Output price per million tokens.

    Returns:
        Cost in USD.
    """
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (completion_tokens / 1_000_000) * output_price_per_million
    return input_cost + output_cost


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string.

    Args:
        s: Input string.
        max_length: Maximum length.
        suffix: Suffix appended after truncation.

    Returns:
        Truncated string.
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def extract_model_name(model: str) -> str:
    """
    Extract the short name from a full model name.

    Example: "gpt-4-turbo-2024-04-09" -> "gpt-4-turbo"

    Args:
        model: Full model name.

    Returns:
        Short model name.
    """
    # Remove date suffix
    parts = model.split('-')
    if len(parts) > 2 and parts[-1].isdigit():
        # Likely a date suffix; strip the trailing date parts
        return '-'.join(parts[:-3]) if len(parts) > 3 else model
    return model


def safe_get(d: dict, *keys, default=None):
    """
    Safely get a value from a nested dict.

    Args:
        d: Dictionary.
        *keys: Key path.
        default: Default value.

    Returns:
        The value found, or the default.

    Example:
        safe_get({"a": {"b": {"c": 1}}}, "a", "b", "c")  # returns 1
        safe_get({"a": {"b": {}}}, "a", "b", "c", default=0)  # returns 0
    """
    result = d
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result
