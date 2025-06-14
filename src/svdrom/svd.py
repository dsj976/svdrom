from abc import ABC, abstractmethod

import dask.array as da
from dask import persist
from dask_ml.decomposition import TruncatedSVD as TSVD

from svdrom.logger import setup_logger

logger = setup_logger("SVD", "svd.log")


class SVD(ABC):
    def __init__(self, X: da.Array) -> None:
        if not isinstance(X, da.Array):
            msg = "The input array must be a dask.array.Array."
            logger.exception(msg)
            raise TypeError(msg)
        if X.ndim != 2:
            msg = "The input array must be two-dimensional."
            logger.exception(msg)
            raise ValueError(msg)
        self.X = X

    def rechunk(self, chunk_cols=False):
        if not chunk_cols:
            self.X = self.X.rechunk({0: "auto", 1: -1})

    @abstractmethod
    def fit(self, n_components: int):
        pass


class ExactSVD(SVD):
    def __init__(self, X):
        super().__init__(X)

    def fit(self, n_components):
        try:
            logger.info("Fitting exact SVD...")
            u, s, v = da.linalg.svd(self.X)
            u, s, v = persist(u, s, v)
            self.u = u[:, :n_components]
            self.s = s[:n_components].compute()
            self.v = v[:n_components, :].compute()
            logger.info("Finished fitting exact SVD.")
        except Exception as e:
            msg = "Failed fitting exact SVD."
            logger.exception(msg)
            raise RuntimeError(msg) from e


class TruncatedSVD(SVD):
    def __init__(self, X):
        super().__init__(X)

    def fit(self, n_components):
        pass


class RandomizedSVD(SVD):
    def __init__(self, X):
        super().__init__(X)

    def fit(self, n_components):
        pass
