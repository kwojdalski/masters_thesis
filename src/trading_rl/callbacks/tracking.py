"""Infrastructure adapter for experiment-level MLflow configuration."""

from __future__ import annotations

import mlflow


class MLflowExperimentTracker:
    """Keep MLflow experiment setup behind an infrastructure boundary."""

    def configure_experiment(
        self, experiment_name: str, tracking_uri: str | None = None
    ) -> str:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        return experiment_name
