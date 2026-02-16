"""Minimal wrapper for optional Weights & Biases logging."""

import os


class WandbLogger:
    """Wrapper around `wandb` for optional experiment tracking."""

    def __init__(self, args, run_id):
        """Create a WandbLogger instance and configure defaults from `args`."""
        self.run_id = run_id
        self.enabled = getattr(args, "use_wandb", False)
        self.project = getattr(args, "wandb_project", "evolvegcn")
        self.entity = getattr(args, "wandb_entity", None)
        self.run_name = getattr(args, "wandb_run_name", None) or run_id
        self.log_minibatch = getattr(args, "wandb_log_minibatch", False)

        if self.enabled:
            import wandb

            self.wandb = wandb
            self.run = None
        else:
            self.wandb = None
            self.run = None

    def init(self, config_dict, config_file_path=None):
        """Initialize wandb run and optionally save config file."""
        if not self.enabled:
            return

        init_kwargs = {
            "project": self.project,
            "name": self.run_name,
            "config": config_dict,
            "reinit": True,
        }
        if self.entity:
            init_kwargs["entity"] = self.entity

        self.run = self.wandb.init(**init_kwargs)

        if config_file_path and os.path.exists(config_file_path):
            self.wandb.save(config_file_path, policy="now")

    def log_epoch_metrics(self, split, epoch, metrics):
        """Log epoch-level `metrics` under `split` in wandb."""
        if not self.enabled or self.run is None:
            return

        log_dict = {"epoch": epoch}
        for key, value in metrics.items():
            log_dict[f"{split}/{key}"] = value

        self.wandb.log(log_dict)

    def log_minibatch_metrics(self, split, epoch, minibatch, metrics):
        """Log minibatch-level `metrics` to wandb if enabled and permitted."""
        if not self.enabled or self.run is None or not self.log_minibatch:
            return

        log_dict = {"epoch": epoch, "minibatch": minibatch}
        for key, value in metrics.items():
            log_dict[f"{split}/{key}"] = value

        self.wandb.log(log_dict)

    def log_artifact(self, path, artifact_type, name, description=None):
        """Log a file `path` as a wandb artifact of type `artifact_type`."""
        if not self.enabled or self.run is None:
            return

        if not os.path.exists(path):
            return

        artifact = self.wandb.Artifact(name=name, type=artifact_type, description=description)
        artifact.add_file(path)
        self.wandb.log_artifact(artifact)

    def save_file(self, path):
        """Save local `path` to wandb storage (if enabled)."""
        if not self.enabled or self.run is None:
            return

        if not os.path.exists(path):
            return

        self.wandb.save(path, policy="now")

    def log_summary(self, key, value):
        """Write a single summary `key`/`value` into wandb run summary."""
        if not self.enabled or self.run is None:
            return

        self.wandb.run.summary[key] = value

    def finish(self):
        """Finish the wandb run if enabled."""
        if not self.enabled or self.run is None:
            return

        self.wandb.finish()
