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
        self._check_matrix_type()

    def _check_matrix_type(self, aspect_ratio=10):
        n_rows, n_cols = self.X.shape
        if (n_rows // n_cols) >= aspect_ratio:
            self.matrix_type = "tall-and-skinny"
        elif (n_cols // n_rows) >= aspect_ratio:
            self.matrix_type = "short-and-fat"
        else:
            self.matrix_type = "square"

    @abstractmethod
    def fit(self, n_components: int):
        pass


class ExactSVD(SVD):
    def __init__(self, X):
        super().__init__(X)
        msg = (
            "Will need to rechunk the array before fitting the SVD. "
            "This will add some overhead."
        )
        if (
            self.matrix_type == "tall-and-skinny"
            and self.X.shape[1] != self.X.chunksize[1]
        ):
            logger.info(msg)
            self.X = self.X.rechunk({0: "auto", 1: -1})
        if (
            self.matrix_type == "short-and-fat"
            and self.X.shape[0] != self.X.chunksize[0]
        ):
            logger.info(msg)
            self.X = self.X.rechunk({0: -1, 1: "auto"})
        if self.matrix_type == "square":
            msg = (
                "The exact SVD algorithm can only handle tall-and-skinny "
                "or short-and-fat matrices, i.e. the aspect ratio must be "
                ">= 10. Try using the randomized SVD algorithm instead."
            )
            logger.exception(msg)
            raise RuntimeError(msg)

    def fit(self, n_components):
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
