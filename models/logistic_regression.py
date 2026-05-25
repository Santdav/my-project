from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_model import BaseChurnModel


class LogisticRegressionChurn(BaseChurnModel):
    """Logistic Regression baseline for churn prediction.

    Wraps sklearn's LogisticRegression in a Pipeline with StandardScaler
    (LR is sensitive to feature scale). Uses class_weight='balanced' to
    handle the ~26% churn imbalance without resampling.

    This model serves as the performance floor — every subsequent model
    (Random Forest, XGBoost) must beat its AUC-ROC to justify added complexity.

    Parameters
    ----------
    C : float
        Inverse of regularization strength. Smaller → stronger L2 penalty.
        Default 1.0 is a reasonable starting point; tune via GridSearchCV
        over [0.01, 0.1, 1.0, 10.0] if needed.
    max_iter : int
        Solver iteration limit. 1000 is safe for this dataset size.
    random_state : int
        Reproducibility seed passed to the solver.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 67, # SIX SEVEEEEN
    ):
        super().__init__(name="LogisticRegression")

        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.C,
                        class_weight="balanced",
                        max_iter=self.max_iter,
                        solver="lbfgs",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    # ------------------------------------------------------------------
    # BaseChurnModel interface
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit scaler + classifier on training data."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"[{self.name}] Training complete — {X_train.shape[0]} samples, "
              f"{X_train.shape[1]} features.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions at default 0.5 threshold."""
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return churn probability (positive class only)."""
        self._check_trained()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        """Persist the full pipeline (scaler + model) via joblib."""
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[{self.name}] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionChurn":
        """Load a saved pipeline and return a ready-to-use instance."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No model file found at {path}")

        instance = cls.__new__(cls)
        super(LogisticRegressionChurn, instance).__init__(name="LogisticRegression")
        instance.model = joblib.load(path)
        instance.is_trained = True
        print(f"[LogisticRegression] Loaded from {path}")
        return instance

    # ------------------------------------------------------------------
    # LR-specific utility
    # ------------------------------------------------------------------

    def get_coefficients(self, feature_names: list[str]) -> dict[str, float]:
        """Return a feature → coefficient mapping for interpretability.

        Useful for sanity-checking against EDA findings — e.g., confirming
        that month-to-month contract and fiber optic carry positive weights.
        """
        self._check_trained()
        classifier = self.model.named_steps["classifier"]
        coefs = classifier.coef_[0]

        if len(feature_names) != len(coefs):
            raise ValueError(
                f"Expected {len(coefs)} feature names, got {len(feature_names)}."
            )

        return dict(sorted(
            zip(feature_names, coefs),
            key=lambda x: abs(x[1]),
            reverse=True,
        ))