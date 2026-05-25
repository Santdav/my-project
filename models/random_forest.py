from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseChurnModel


class RandomForestChurn(BaseChurnModel):
    """Random Forest model for churn prediction.

    Parameters
    ----------
    n_estimators : int
        Number of trees. 300 is a solid default — diminishing returns
        beyond ~500 on a dataset this size.
    max_depth : int | None
        Maximum tree depth. None = fully grown trees (can overfit).
        Tune over [None, 10, 20, 30] if validation performance lags.
    min_samples_leaf : int
        Minimum samples required at a leaf node. Acts as implicit
        regularization — higher values → smoother decision boundary.
    max_features : str
        Features considered at each split. 'sqrt' is the standard
        default for classification tasks.
    random_state : int
        Reproducibility seed.
    n_jobs : int
        Parallel jobs for fitting. -1 uses all available cores.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(name="RandomForest")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    # ------------------------------------------------------------------
    # BaseChurnModel interface
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the forest on training data."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(
            f"[{self.name}] Training complete — {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features, {self.n_estimators} trees."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions at default 0.5 threshold."""
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return churn probability (positive class only)."""
        self._check_trained()
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        """Persist the trained forest via joblib."""
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[{self.name}] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestChurn":
        """Load a saved model and return a ready-to-use instance."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No model file found at {path}")

        instance = cls.__new__(cls)
        super(RandomForestChurn, instance).__init__(name="RandomForest")
        instance.model = joblib.load(path)
        instance.is_trained = True
        print(f"[RandomForest] Loaded from {path}")
        return instance

    # ------------------------------------------------------------------
    # RF-specific utilities
    # ------------------------------------------------------------------

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Return a DataFrame of features ranked by mean decrease in impurity.

        This gives you a quick signal on which engineered features pulled
        weight — confirm whether MultipleLines / PhoneService are dead weight,
        and validate that risk_score and contract type rank highly.

        Returns
        -------
        pd.DataFrame
            Columns: ['feature', 'importance'], sorted descending.
        """
        self._check_trained()
        importances = self.model.feature_importances_

        if len(feature_names) != len(importances):
            raise ValueError(
                f"Expected {len(importances)} feature names, got {len(feature_names)}."
            )

        return (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_oob_score(self) -> float:
        """Return the out-of-bag score if available.

        OOB score is a free internal validation estimate — useful as a
        sanity check before running full cross-validation.
        Note: requires oob_score=True at construction time.
        """
        self._check_trained()
        if not hasattr(self.model, "oob_score_"):
            raise AttributeError(
                "OOB score not available. Re-instantiate with oob_score=True."
            )
        return self.model.oob_score_