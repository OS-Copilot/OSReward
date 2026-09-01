import re
from typing import Any


def parse_cua_gym_reward(actual: str, expected: Any = None, **options: Any) -> float:
    """Parse the numeric score printed by a CUA-Gym reward.py script."""
    _ = expected, options
    if actual is None:
        return 0.0

    matches = re.findall(r"REWARD:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", str(actual))
    if not matches:
        return 0.0

    try:
        score = float(matches[-1])
    except ValueError:
        return 0.0

    return max(0.0, min(1.0, score))
