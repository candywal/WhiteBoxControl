# white_box_monitoring/evaluations/metrics.py
from dataclasses import dataclass
from typing import List, Any


@dataclass
class EvaluationMetrics:
    """
    Evaluations can have other metrics but these are required.
    """

    monitor_predictions: List[Any]  # experiment config, monitor predictions
