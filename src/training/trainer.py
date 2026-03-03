"""Main training loop for the dual encoder retrieval model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.contrastive import CLIPContrastiveLoss, ContrastiveLossWithHardNegatives
from src.losses.hard_negative import HardNegativeMiner, build_hard_negative_miner
from src.losses.triplet import TripletLoss
from src.models.dual_encoder import DualEncoder
from src.training.callbacks import EarlyStopping, MetricsLogger
from src.training.checkpointing import CheckpointManager
from src.training.mixed_precision import MixedPrecisionManager
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.utils.logging import get_logger


class Trainer:
    """Professional training loop for the dual encoder model.

    Supports:
        - Mixed precision training (fp16/bf16)
        - Gradient clipping and accumulation
        - Differential learning rates per component
        - Hard negative mining (epoch-gated)
        - Early stopping + top-k checkpointing
        - WandB / TensorBoard logging
        - Comprehensive metric tracking

    Args:
        model: DualEncoder model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        cfg: Merged configuration dictionary.
        device: Torch device.
    """

    def __init__(
        self,
        model: DualEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.logger = get_logger("trainer")

        # ── Training config ──
        train_cfg = cfg.get("train", {})
        loss_cfg = cfg.get("loss", {})

        self.epochs = train_cfg.get("epochs", 20)
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
        self.log_every_n = cfg.get("logging", {}).get("log_every_n_steps", 50)

        # ── Output directory ──
        project_cfg = cfg.get("project", {})
        self.output_dir = Path(
            project_cfg.get("output_dir", "outputs")
        ) / project_cfg.get("experiment_name", "experiment")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Optimizer ──
        param_groups = model.get_param_groups(
            lr_image=train_cfg.get("lr_image", train_cfg.get("lr", 1e-5)),
            lr_text=train_cfg.get("lr_text", train_cfg.get("lr", 1e-5)),
            lr_projection=train_cfg.get("lr_projection", train_cfg.get("lr", 1e-4)),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
        self.optimizer = build_optimizer(
            param_groups,
            optimizer_name=train_cfg.get("optimizer", "adamw"),
        )

        # ── Scheduler ──
        total_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_name=train_cfg.get("scheduler", "cosine"),
            num_training_steps=total_steps,
            num_warmup_steps=train_cfg.get("warmup_steps", 0),
        )

        # ── Loss function ──
        self.loss_fn = self._build_loss(loss_cfg)

        # ── Hard negative miner ──
        hn_cfg = loss_cfg.get("hard_negative_mining", {})
        self.hard_negative_miner = build_hard_negative_miner(hn_cfg)

        # ── Mixed precision ──
        device_cfg = cfg.get("device", {})
        self.amp_manager = MixedPrecisionManager(
            enabled=device_cfg.get("mixed_precision", "fp16") != "off",
            dtype=device_cfg.get("mixed_precision", "fp16"),
            device_type=device.type,
        )

        # ── Checkpointing ──
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.output_dir,
            save_top_k=cfg.get("logging", {}).get("save_top_k", 3),
            monitor_metric=train_cfg.get("early_stopping_metric", "val_r_at_10_t2i"),
            mode=train_cfg.get("early_stopping_mode", "max"),
        )

        # ── Early stopping ──
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 5),
            metric=train_cfg.get("early_stopping_metric", "val_r_at_10_t2i"),
            mode=train_cfg.get("early_stopping_mode", "max"),
        )

        # ── Metrics logger ──
        self.metrics_logger = MetricsLogger()

        # ── WandB (optional) ──
        self.wandb_run = None
        if cfg.get("logging", {}).get("use_wandb", False):
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=cfg.get("logging", {}).get("project_name", "retrieval"),
                    name=project_cfg.get("experiment_name", "experiment"),
                    config=dict(cfg) if hasattr(cfg, '__iter__') else {},
                )
            except Exception as e:
                self.logger.warning(f"WandB init failed: {e}")

        self.global_step = 0

    def _build_loss(self, loss_cfg: dict) -> nn.Module:
        """Build the loss function from config."""
        loss_name = loss_cfg.get("name", "clip_symmetric_infonce")

        if loss_name == "clip_symmetric_infonce":
            if loss_cfg.get("hard_negative_mining", {}).get("enabled", False):
                return ContrastiveLossWithHardNegatives(
                    label_smoothing=loss_cfg.get("label_smoothing", 0.0),
                )
            else:
                return CLIPContrastiveLoss(
                    label_smoothing=loss_cfg.get("label_smoothing", 0.0),
                )
        elif loss_name == "triplet":
            return TripletLoss(
                margin=loss_cfg.get("margin", 0.2),
            )
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary with final training results and best metrics.
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Model info: {self.model.get_model_info()}")

        best_metrics = {}

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # ── Train epoch ──
            train_metrics = self._train_epoch(epoch)

            # ── Validate epoch ──
            val_metrics = self._validate_epoch(epoch)

            # ── Merge metrics ──
            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            epoch_time = time.time() - epoch_start
            all_metrics["epoch_time_s"] = epoch_time

            self.metrics_logger.log_epoch(all_metrics)

            # ── Logging ──
            self.logger.info(
                f"Epoch {epoch}/{self.epochs-1} | "
                f"train_loss: {train_metrics.get('train_loss', 0):.4f} | "
                f"val_loss: {val_metrics.get('val_loss', 0):.4f} | "
                f"val_r@10_t2i: {val_metrics.get('val_r_at_10_t2i', 0):.4f} | "
                f"time: {epoch_time:.1f}s"
            )

            if self.wandb_run:
                import wandb
                wandb.log(all_metrics, step=epoch)

            # ── Checkpointing ──
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=all_metrics,
            )

            # ── Early stopping ──
            if self.early_stopping(val_metrics):
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best score: {self.early_stopping.best_score:.4f}"
                )
                break

        # ── Final results ──
        best = self.metrics_logger.get_best_epoch(
            self.early_stopping.metric,
            self.early_stopping.mode,
        )
        self.logger.info(f"Training complete. Best epoch: {best}")

        if self.wandb_run:
            self.wandb_run.finish()

        return {
            "best_epoch": best,
            "final_metrics": all_metrics,
            "output_dir": str(self.output_dir),
        }

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch.

        Returns:
            Dictionary of averaged training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_loss_i2t = 0.0
        total_loss_t2i = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch}",
            leave=False,
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Forward pass with mixed precision
            with self.amp_manager.autocast_context():
                outputs = self.model(images, input_ids, attention_mask)

                # Hard negative mining
                hn_mask = None
                if self.hard_negative_miner is not None:
                    hn_mask = self.hard_negative_miner.mine(
                        outputs["image_embeddings"].detach(),
                        outputs["text_embeddings"].detach(),
                        current_epoch=epoch,
                        languages=batch.get("languages"),
                    )

                # Compute loss
                if isinstance(self.loss_fn, (CLIPContrastiveLoss, ContrastiveLossWithHardNegatives)):
                    if hn_mask is not None and isinstance(self.loss_fn, ContrastiveLossWithHardNegatives):
                        loss_dict = self.loss_fn(
                            outputs["logits_per_image"],
                            outputs["logits_per_text"],
                            hard_negative_mask=hn_mask,
                        )
                    else:
                        loss_dict = self.loss_fn(
                            outputs["logits_per_image"],
                            outputs["logits_per_text"],
                        )
                elif isinstance(self.loss_fn, TripletLoss):
                    loss_dict = self.loss_fn(
                        outputs["image_embeddings"],
                        outputs["text_embeddings"],
                    )
                else:
                    loss_dict = self.loss_fn(
                        outputs["logits_per_image"],
                        outputs["logits_per_text"],
                    )

                loss = loss_dict["loss"] / self.grad_accum_steps

            # Backward pass
            scaled_loss = self.amp_manager.scale_loss(loss)
            scaled_loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.amp_manager.unscale_gradients(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
                self.amp_manager.step(self.optimizer)
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Track metrics
            total_loss += loss_dict["loss"].item()
            total_loss_i2t += loss_dict.get("loss_i2t", loss_dict["loss"]).item()
            total_loss_t2i += loss_dict.get("loss_t2i", loss_dict["loss"]).item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['loss'].item():.4f}",
                "temp": f"{outputs['temperature'].item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Step-level logging
            if self.global_step % self.log_every_n == 0:
                step_metrics = {
                    "train_step_loss": loss_dict["loss"].item(),
                    "temperature": outputs["temperature"].item(),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "global_step": self.global_step,
                }
                self.metrics_logger.log_step(step_metrics)

        avg_loss = total_loss / max(num_batches, 1)
        return {
            "train_loss": avg_loss,
            "train_loss_i2t": total_loss_i2t / max(num_batches, 1),
            "train_loss_t2i": total_loss_t2i / max(num_batches, 1),
        }

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one validation epoch.

        Computes loss + retrieval metrics (R@1, R@5, R@10) on validation set.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_image_embeddings = []
        all_text_embeddings = []

        pbar = tqdm(
            self.val_loader,
            desc=f"Val Epoch {epoch}",
            leave=False,
        )

        for batch in pbar:
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with self.amp_manager.autocast_context():
                outputs = self.model(images, input_ids, attention_mask)

                if isinstance(self.loss_fn, (CLIPContrastiveLoss, ContrastiveLossWithHardNegatives)):
                    loss_dict = self.loss_fn(
                        outputs["logits_per_image"],
                        outputs["logits_per_text"],
                    )
                elif isinstance(self.loss_fn, TripletLoss):
                    loss_dict = self.loss_fn(
                        outputs["image_embeddings"],
                        outputs["text_embeddings"],
                    )
                else:
                    loss_dict = self.loss_fn(
                        outputs["logits_per_image"],
                        outputs["logits_per_text"],
                    )

            total_loss += loss_dict["loss"].item()
            num_batches += 1

            # Collect embeddings for retrieval metrics
            all_image_embeddings.append(outputs["image_embeddings"].cpu())
            all_text_embeddings.append(outputs["text_embeddings"].cpu())

        avg_loss = total_loss / max(num_batches, 1)

        # ── Compute retrieval metrics ──
        all_image_emb = torch.cat(all_image_embeddings, dim=0)
        all_text_emb = torch.cat(all_text_embeddings, dim=0)

        retrieval_metrics = self._compute_retrieval_metrics(all_image_emb, all_text_emb)

        return {
            "val_loss": avg_loss,
            **retrieval_metrics,
        }

    def _compute_retrieval_metrics(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        ks: list[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """Compute R@K retrieval metrics.

        Args:
            image_embeddings: (N, D) image embeddings.
            text_embeddings: (N, D) text embeddings.
            ks: List of K values for Recall@K.

        Returns:
            Dictionary with val_r_at_K_t2i and val_r_at_K_i2t metrics.
        """
        n = image_embeddings.shape[0]
        sim_matrix = image_embeddings @ text_embeddings.T
        targets = torch.arange(n)

        metrics = {}

        for k in ks:
            # Text → Image: for each text, find matching image
            _, t2i_topk = sim_matrix.T.topk(k, dim=1)
            t2i_correct = (t2i_topk == targets.unsqueeze(1)).any(dim=1).float()
            metrics[f"val_r_at_{k}_t2i"] = t2i_correct.mean().item()

            # Image → Text: for each image, find matching text
            _, i2t_topk = sim_matrix.topk(k, dim=1)
            i2t_correct = (i2t_topk == targets.unsqueeze(1)).any(dim=1).float()
            metrics[f"val_r_at_{k}_i2t"] = i2t_correct.mean().item()

        return metrics
