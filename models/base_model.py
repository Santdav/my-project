# models/base_model.py
from abc import ABC

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


class BaseChurnModel(ABC):
    """Abstract base class for all churn prediction models.

    Every model must implement train, predict, predict_proba, save, and load.
    evaluate() is provided here since the metric contract is shared across models.
    """

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False

    # ------------------------------------------------------------------
    # Abstract interface — each subclass must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model on training data."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 / 1)."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return churn probability for each sample (positive class)."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the trained model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseChurnModel":
        """Load a persisted model from disk and return an instance."""
        ...

    # ------------------------------------------------------------------
    # Shared evaluation — same metric contract for all models
    # ------------------------------------------------------------------

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Compute and return the standard metric suite.

        Primary:   AUC-ROC
        Secondary: Average Precision (PR-AUC)
        Diagnostic: confusion matrix, full classification report
        """
        self._check_trained()

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        metrics = {
            "auc_roc": roc_auc_score(y_test, y_proba),
            "avg_precision": average_precision_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["No Churn", "Churn"]
            ),
        }
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                f"{self.name} has not been trained yet. Call train() first."
            )

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.name}', status={status})"