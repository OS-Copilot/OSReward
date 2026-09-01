from __future__ import annotations
from typing import Any, Dict
import logging

logger = logging.getLogger("desktopenv.oscaliber.metrics")


def oscaliber_rule_based_metric(result: Any, expected: Any, options: Dict[str, Any] | None = None) -> float:
    """Rule-based 版本评估 metric 的骨架.

    - result/expected: 来自 file/terminal 等 getter 的结构化结果
    - options["oscaliber_meta"]: 在生成任务时注入的元数据, 包含
        - evaluation_type
        - evaluation_desc
        - condition
        - expected_result
        - complexity
        - estimated_steps
        - category

    返回值约定:
      - 1.0: 通过
      - 0.0: 未通过
      - 可扩展为 [0,1] 区间的部分得分
    """
    options = options or {}
    meta = options.get("oscaliber_meta", {})
    logger.debug("[oscaliber_rule_based_metric] meta=%s, result=%s, expected=%s", meta, result, expected)

    # TODO: 在这里根据 condition / expected_result + result/expected 实现真正的规则判定
    # 目前作为骨架, 统一返回 0.0
    return 0.0


