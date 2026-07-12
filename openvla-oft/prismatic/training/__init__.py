"""Training exports with lazy imports.

Inference imports ``prismatic.training.train_utils`` for action-mask helpers.
Importing this package must not eagerly load the training metrics stack (and its
optional Weights & Biases dependency) during an offline evaluation.
"""

__all__ = ["get_train_strategy", "Metrics", "VLAMetrics"]


def __getattr__(name):
    if name == "get_train_strategy":
        from .materialize import get_train_strategy

        return get_train_strategy
    if name in {"Metrics", "VLAMetrics"}:
        from .metrics import Metrics, VLAMetrics

        return {"Metrics": Metrics, "VLAMetrics": VLAMetrics}[name]
