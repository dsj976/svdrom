from abc import ABC, abstractmethod

import dask.array as da
from dask import persist

from svdrom.logger import setup_logger

logger = setup_logger("SVD", "svd.log")


class SVD(ABC):
    """
    Abstract base class for performing Singular Value Decomposition (SVD) on a
    two-dimensional Dask array.

    Parameters
    ----------
    X : da.Array
        The input two-dimensional Dask array to decompose.

    Attributes
    ----------
    X : da.Array
        The input data matrix.
    matrix_type : str
        The type of the matrix based on its aspect ratio: "tall-and-skinny",
        "short-and-fat" or "square".

    Methods
    -------
    fit(n_components: int)
        Abstract method to fit the SVD model to the input data, retaining
        the specified number of components.

    Raises
    ------
    ValueError
        If the input array is not two-dimensional.
    """

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
    """
    ExactSVD performs an exact Singular Value Decomposition (SVD)
    on a Dask array.

    This class inherits from the SVD base class and implements
    the exact SVD algorithm for matrices that are either
    tall-and-skinny or short-and-fat, with an aspect ratio >= 10.
    It rechunks the input array as needed to optimize SVD
    computation and raises an exception if the matrix is square,
    recommending the use of randomized SVD instead.

    Parameters
    ----------
    X : dask.array.Array
        The input matrix to decompose.

    Attributes
    ----------
    u : dask.array.Array
        Left singular vectors, shape (n_samples, n_components).
    s : numpy.ndarray
        Singular values, shape (n_components,).
    v : dask.array.Array
        Right singular vectors, shape (n_components, n_features).

    Methods
    -------
    fit(n_components)
        Computes the exact SVD and stores the top `n_components`
        left singular vectors (`u`), singular values (`s`),
        and right singular vectors (`v`).

    Raises
    ------
    RuntimeError
        If the input matrix is square/nearly square or if SVD computation fails.
    """

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
        msg = (
            "Rechunking the array before fitting the randomized SVD. "
            "This might add some overhead."
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

    def fit(self, n_components):
        self.X = self.X.persist()
        try:
            logger.info("Fitting randomized SVD...")
            u, s, v = da.linalg.svd_compressed(self.X, n_components)
            u, s, v = persist(u, s, v)
            self.u = u[:, :n_components]
            self.v = v[:n_components, :]
            self.s = s[:n_components].compute()
            logger.info("Finished fitting randomized SVD.")
        except Exception as e:
            msg = "Failed fitting randomized SVD."
            logger.exception(msg)
            raise RuntimeError(msg) from e
