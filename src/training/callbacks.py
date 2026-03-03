"""Training callbacks for logging, early stopping, etc."""

from __future__ import annotations

from typing import Any, Dict, Optional


class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving.

    Args:
        patience: Number of epochs without improvement before stopping.
        metric: Metric name to monitor.
        mode: 'max' (higher is better) or 'min' (lower is better).
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 5,
        metric: str = "val_r_at_10_t2i",
        mode: str = "max",
        min_delta: float = 0.001,
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta

        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, metrics: Dict[str, float]) -> bool:
        """Check if training should stop.

        Args:
            metrics: Dictionary of current validation metrics.

        Returns:
            True if training should stop.
        """
        score = metrics.get(self.metric)
        if score is None:
            return False

        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience
        return self.should_stop

    def get_status(self) -> Dict[str, Any]:
        """Return current early stopping status."""
        return {
            "best_score": self.best_score,
            "counter": self.counter,
            "patience": self.patience,
            "should_stop": self.should_stop,
        }


class MetricsLogger:
    """Accumulates and tracks metrics across training steps and epochs."""

    def __init__(self):
        self.step_metrics: list[Dict[str, float]] = []
        self.epoch_metrics: list[Dict[str, float]] = []
        self.current_step_accumulator: Dict[str, list] = {}

    def log_step(self, metrics: Dict[str, float]) -> None:
        """Log metrics for a single training step."""
        self.step_metrics.append(metrics)
        for key, value in metrics.items():
            if key not in self.current_step_accumulator:
                self.current_step_accumulator[key] = []
            self.current_step_accumulator[key].append(value)

    def log_epoch(self, metrics: Dict[str, float]) -> None:
        """Log metrics for a completed epoch."""
        self.epoch_metrics.append(metrics)
        self.current_step_accumulator.clear()

    def get_epoch_average(self) -> Dict[str, float]:
        """Compute average of accumulated step metrics."""
        averages = {}
        for key, values in self.current_step_accumulator.items():
            averages[key] = sum(values) / len(values) if values else 0.0
        return averages

    def get_best_epoch(self, metric: str, mode: str = "max") -> Dict[str, Any]:
        """Find the best epoch by a specific metric."""
        if not self.epoch_metrics:
            return {}

        if mode == "max":
            best_idx = max(
                range(len(self.epoch_metrics)),
                key=lambda i: self.epoch_metrics[i].get(metric, float("-inf")),
            )
        else:
            best_idx = min(
                range(len(self.epoch_metrics)),
                key=lambda i: self.epoch_metrics[i].get(metric, float("inf")),
            )

        return {
            "epoch": best_idx,
            "metrics": self.epoch_metrics[best_idx],
        }
