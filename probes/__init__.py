from .logistic_probe import LogisticProbe
from .mean_diff_probe import MeanDiffProbe
from .attention_probe import LearnedAttentionProbe

PROBE_CLASSES = {
    LogisticProbe.name: LogisticProbe,
    MeanDiffProbe.name: MeanDiffProbe,
    LearnedAttentionProbe.name: LearnedAttentionProbe,
}

def get_probe_class(probe_type_str: str):
    if probe_type_str not in PROBE_CLASSES:
        raise ValueError(f"Unknown probe type: {probe_type_str}. Available: {list(PROBE_CLASSES.keys())}")
    return PROBE_CLASSES[probe_type_str]