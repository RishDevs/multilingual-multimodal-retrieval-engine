"""Model checkpointing with best-model tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """Manages model checkpointing with best-model tracking.

    Saves top-k checkpoints by a monitored metric, plus the latest
    checkpoint. Also saves the merged config for reproducibility.

    Args:
        output_dir: Directory to save checkpoints.
        save_top_k: Number of best checkpoints to keep.
        monitor_metric: Metric name to monitor for best model selection.
        mode: 'max' or 'min' — whether higher is better.
    """

    def __init__(
        self,
        output_dir: str | Path,
        save_top_k: int = 3,
        monitor_metric: str = "val_r_at_10_t2i",
        mode: str = "max",
    ):
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor_metric = monitor_metric
        self.mode = mode

        self.best_scores: list[tuple[float, Path]] = []

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Save a checkpoint and manage best-model tracking.

        Args:
            model: Model to save.
            optimizer: Optimizer state to save.
            epoch: Current epoch number.
            metrics: Dictionary of validation metrics.
            config: Optional config dict for reproducibility.

        Returns:
            Path to saved checkpoint, or None if not saved (not in top-k).
        """
        score = metrics.get(self.monitor_metric, 0.0)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "monitor_metric": self.monitor_metric,
            "score": score,
        }

        if config:
            checkpoint["config"] = config

        # Always save latest
        latest_path = self.output_dir / "latest.ckpt"
        torch.save(checkpoint, str(latest_path))

        # Check if this is a top-k checkpoint
        save_path = self.output_dir / f"epoch_{epoch:03d}_score_{score:.4f}.ckpt"
        should_save = len(self.best_scores) < self.save_top_k

        if not should_save:
            if self.mode == "max":
                worst_score = min(s for s, _ in self.best_scores)
                should_save = score > worst_score
            else:
                worst_score = max(s for s, _ in self.best_scores)
                should_save = score < worst_score

        if should_save:
            torch.save(checkpoint, str(save_path))
            self.best_scores.append((score, save_path))

            # Remove worst if exceeding top_k
            if len(self.best_scores) > self.save_top_k:
                if self.mode == "max":
                    self.best_scores.sort(key=lambda x: x[0], reverse=True)
                else:
                    self.best_scores.sort(key=lambda x: x[0])

                _, remove_path = self.best_scores.pop()
                if remove_path.exists():
                    remove_path.unlink()

            # Update best symlink
            if self.mode == "max":
                best = max(self.best_scores, key=lambda x: x[0])
            else:
                best = min(self.best_scores, key=lambda x: x[0])

            best_link = self.output_dir / "best.ckpt"
            if best_link.exists():
                best_link.unlink()
            torch.save(torch.load(str(best[1])), str(best_link))

            return save_path

        return None

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state.
            map_location: Device mapping for loading.

        Returns:
            Checkpoint metadata dict (epoch, metrics, etc.).
        """
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "score": checkpoint.get("score", 0.0),
        }
