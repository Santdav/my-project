from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .base_model import BaseChurnModel


class XGBoostChurn(BaseChurnModel):
    """XGBoost model for churn prediction.

    Gradient boosted trees — the expected top performer in this pipeline.
    Handles imbalance via scale_pos_weight instead of class_weight, which
    directly scales the gradient of the minority class during boosting.

    No scaling needed (tree-based). Supports early stopping to avoid
    overfitting without requiring a manual n_estimators grid search.

    Parameters
    ----------
    n_estimators : int
        Maximum number of boosting rounds. With early stopping enabled,
        the actual number used will typically be lower.
    max_depth : int
        Maximum tree depth. Shallower trees (3–6) generalize better for
        tabular data — deeper trees risk memorizing noise.
    learning_rate : float
        Step size shrinkage. Lower rate + more rounds = better generalization
        but slower training. 0.05–0.1 is a good starting range.
    subsample : float
        Fraction of training samples used per boosting round. Acts as
        stochastic regularization, similar to RF's bootstrap.
    colsample_bytree : float
        Fraction of features sampled per tree. Reduces correlation between
        trees, analogous to RF's max_features='sqrt'.
    scale_pos_weight : float | None
        Weight applied to the positive (churn) class gradient.
        If None, computed automatically as neg_count / pos_count (~2.85
        for this dataset's ~26% churn rate). Pass an explicit value to
        override after threshold analysis if recall is still poor.
    early_stopping_rounds : int | None
        Stop boosting if validation AUC doesn't improve for this many rounds.
        Requires passing eval_set to train(). Set None to disable.
    random_state : int
        Reproducibility seed.
    n_jobs : int
        Parallel jobs. -1 uses all available cores.
    """

    # Approximate class ratio for IBM Telco dataset (~74% no-churn / ~26% churn)
    _DEFAULT_SCALE_POS_WEIGHT = 74 / 26

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float | None = None,
        early_stopping_rounds: int | None = 30,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(name="XGBoost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = (
            scale_pos_weight
            if scale_pos_weight is not None
            else self._DEFAULT_SCALE_POS_WEIGHT
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    # ------------------------------------------------------------------
    # BaseChurnModel interface
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Fit the booster on training data.

        Parameters
        ----------
        X_train, y_train : training split
        X_val, y_val : optional validation split for early stopping.
            If not provided and early_stopping_rounds is set, a warning
            is printed and early stopping is skipped.
        """
        eval_set = None

        if self.early_stopping_rounds is not None:
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
            else:
                print(
                    f"[{self.name}] Warning: early_stopping_rounds={self.early_stopping_rounds} "
                    "but no validation set provided — early stopping disabled."
                )

        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = 50  # log every 50 rounds

        self.model.fit(X_train, y_train, **fit_params)
        self.is_trained = True

        best = getattr(self.model, "best_iteration", None)
        rounds_used = best + 1 if best is not None else self.n_estimators
        print(
            f"[{self.name}] Training complete — {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features, {rounds_used} rounds used."
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
        """Persist the trained booster via joblib."""
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[{self.name}] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostChurn":
        """Load a saved model and return a ready-to-use instance."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No model file found at {path}")

        instance = cls.__new__(cls)
        super(XGBoostChurn, instance).__init__(name="XGBoost")
        instance.model = joblib.load(path)
        instance.is_trained = True
        print(f"[XGBoost] Loaded from {path}")
        return instance

    # ------------------------------------------------------------------
    # XGBoost-specific utilities
    # ------------------------------------------------------------------

    def get_feature_importance(
        self,
        feature_names: list[str],
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """Return a DataFrame of features ranked by importance.

        Parameters
        ----------
        importance_type : str
            One of 'gain', 'weight', 'cover'.
            - 'gain'   : average loss reduction per split (most informative)
            - 'weight' : number of times feature is used in splits
            - 'cover'  : average number of samples affected per split

        Returns
        -------
        pd.DataFrame
            Columns: ['feature', 'importance'], sorted descending.
        """
        self._check_trained()
        self.model.get_booster().feature_names = feature_names

        raw = self.model.get_booster().get_score(importance_type=importance_type)

        # Features with zero splits won't appear in get_score — fill them
        importance = {f: raw.get(f, 0.0) for f in feature_names}

        return (
            pd.DataFrame(
                {"feature": list(importance.keys()), "importance": list(importance.values())}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_best_iteration(self) -> int | None:
        """Return the best boosting round found by early stopping, if used."""
        self._check_trained()
        return getattr(self.model, "best_iteration", None)