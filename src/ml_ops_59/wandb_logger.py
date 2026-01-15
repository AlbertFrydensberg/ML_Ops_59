"""
Weights & Biases (WandB) integration for experiment tracking
"""

from typing import Dict, Optional, Any

import numpy as np

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore


class WandBLogger:
    """
    WandB logger that can be used for both:
    - normal runs (no sweep)
    - sweep runs (wandb.agent / wandb sweep)
    """

    def __init__(
        self,
        project_name: str = "MLops_59",
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled and WANDB_AVAILABLE)
        self._started_run = False  # did THIS logger call wandb.init?

        if not self.enabled:
            self.run = None
            return

        # If a run already exists (e.g., created by a sweep agent), reuse it.
        if wandb.run is None:
            wandb.init(project=project_name, config=config)
            self._started_run = True
        else:
            # Still record config if provided
            if config:
                wandb.config.update(config, allow_val_change=True)

        self.run = wandb.run

        self.config = dict(wandb.config)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_config(self, cfg: Dict[str, Any]) -> None:
        """Add/override config values for this run."""
        if self.enabled:
            wandb.config.update(cfg, allow_val_change=True)

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[list] = None,
    ) -> None:
        if not self.enabled:
            return

        try:
            wandb.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true,
                        preds=y_pred,
                        class_names=class_names,
                    )
                }
            )
        except Exception:
            # Very simple fallback
            try:
                from sklearn.metrics import confusion_matrix as sklearn_cm

                cm = sklearn_cm(y_true, y_pred)
                wandb.log(
                    {
                        "confusion_matrix_data": wandb.Table(
                            data=cm.tolist(),
                            columns=class_names
                            if class_names
                            else [f"Class {i}" for i in range(cm.shape[1])],
                        )
                    }
                )
            except Exception as e:
                print(f"Warning: Could not log confusion matrix to WandB: {e}")

    def log_image(self, name: str, image_path: str) -> None:
        if self.enabled:
            wandb.log({name: wandb.Image(image_path)})

    def log_model_architecture(self, info: Any) -> None:
        """
        Log model info in config.
        - If you pass a dict, it updates config with that dict.
        - If you pass a string, it stores it under 'model_architecture'.
        """
        if not self.enabled:
            return

        if isinstance(info, dict):
            wandb.config.update(info, allow_val_change=True)
        else:
            wandb.config.update({"model_architecture": str(info)}, allow_val_change=True)

    def finish(self) -> None:
        """Finish the WandB run (only if this logger started it)."""
        if self.enabled and self._started_run:
            wandb.finish()
