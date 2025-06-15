from abc import ABC, abstractmethod

import dask.array as da
from dask import persist

from svdrom.logger import setup_logger

logger = setup_logger("SVD", "svd.log")


class SVD(ABC):
    def __init__(self, X: da.Array) -> None:
        if X.ndim != 2:
            msg = "The input array must be two-dimensional."
            logger.exception(msg)
            raise ValueError(msg)
        self.X = X

    def _rechunk(self, col_chunk_size: int = -1):
        self.X = self.X.rechunk({0: "auto", 1: col_chunk_size}).persist()

    @abstractmethod
    def fit(self, n_components: int):
        pass


class ExactSVD(SVD):
    def __init__(self, X):
        super().__init__(X)

    def fit(self, n_components):
        if self.X.shape[1] != self.X.chunksize[1]:
            msg = (
                "Will need to rechunk the array before fitting the SVD. "
                "This will add some overhead."
            )
            logger.info(msg)
            self._rechunk()
        self.X = self.X.persist()
        try:
            logger.info("Fitting exact SVD...")
            u, s, v = da.linalg.svd(self.X)
            u, s, v = persist(u, s, v)
            self.u = u[:, :n_components]
            self.v = v[:n_components, :]
            self.s = s[:n_components].compute()
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
