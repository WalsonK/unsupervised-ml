"""
Minimal PCA implementation based on the singular–value decomposition.

Author : you
Licence: public domain
"""

import numpy as np


class PCA:
    """
    Principal-Component Analysis (truncated to `n_components`).

    Parameters
    ----------
    n_components : int
        How many principal directions to keep.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components

        # learned during fit -------------------------------------------
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None          # (k, D)
        self.explained_variance_: np.ndarray | None = None  # (k,)
        self.singular_values_: np.ndarray | None = None     # (k,)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "PCA":
        """
        Learn the principal directions.

        X : array shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]

        # ----- 1. centre the data -------------------------------------
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # ----- 2. SVD --------------------------------------------------
        # X_centered = U S V^T   (U: n×n,  S: n,  V: n×D)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # ----- 3. keep the first k columns of V^T ----------------------
        k = self.n_components
        self.components_ = Vt[:k]                      # (k, D)
        self.singular_values_ = S[:k]                  # (k,)
        self.explained_variance_ = (S[:k] ** 2) / (n_samples - 1)

        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project `X` onto the previously learned components.
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA.transform() called before fit()")

        X_centered = np.asarray(X, dtype=np.float32) - self.mean_
        return X_centered @ self.components_.T          # (n_samples, k)

    # ------------------------------------------------------------------
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience = fit() + transform().
        """
        return self.fit(X).transform(X)