from abc import ABC, abstractmethod

import dask.array as da
import numpy as np
from dask import persist
from dask_ml.decomposition import TruncatedSVD as tsvd

from svdrom.logger import setup_logger

logger = setup_logger("SVD", "svd.log")


class SVD(ABC):
    """Abstract base class for performing Singular Value Decomposition (SVD) on a
    two-dimensional Dask array. The array is rechunked automatically for an optimized
    SVD computation.

    Attributes
    ----------
    X (dask.array.Array): The input data matrix.
    matrix_type (str): The type of the matrix based on its aspect ratio:
    "tall-and-skinny", "short-and-fat" or "square".
    n_components (int): number of SVD components
    u (dask.array.Array): left singular vectors
    s (numpy.ndarray): singular values
    v (dask.array.Array): right singular vectors
    """

    def __init__(self, X: da.Array) -> None:
        """Initializes a SVD object.

        Parameters
        ----------
        X (dask.array.Array): The input data matrix.
        """
        self.n_components = 0
        self.u = da.empty_like(0)
        self.s = np.empty_like(0)
        self.v = da.empty_like(0)
        if X.ndim != 2:
            msg = "The input array must be two-dimensional."
            logger.exception(msg)
            raise ValueError(msg)
        self.X = X
        self.matrix_type = ""
        self._check_matrix_type()
        self._rechunk_array()

    def _check_matrix_type(self, aspect_ratio=10):
        """Checks if input matrix is tall-and-skinny,
        short-and-fat or square/nearly-square, based on
        the specified aspect ratio.

        Parameters
        ----------
        aspect_ratio (int): defines the matrix type based on the
        ratio between number of rows and columns. Defaults to 10.
        """
        n_rows, n_cols = self.X.shape
        if (n_rows // n_cols) >= aspect_ratio:
            self.matrix_type = "tall-and-skinny"
        elif (n_cols // n_rows) >= aspect_ratio:
            self.matrix_type = "short-and-fat"
        else:
            self.matrix_type = "square"

    def _rechunk_array(self):
        """Rechunks the input array to ensure optimal chunk sizes
        for SVD computation based on matrix type.
        """
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

    @abstractmethod
    def fit(self, n_components: int, transform: bool = False, **kwargs):
        """Perform the SVD fit operation.

        Parameters
        ----------
        n_components (int): number of SVD components to keep.
        transform (bool): whether to compute `u` for a tall-and-skinny
        matrix or `v` for a short-and-fat matrix, since these can be much
        larger than the other SVD results. Defaults to False.
        **kwargs: keyword arguments for the underlying SVD computation.
        """

    @abstractmethod
    def transform(
        self,
    ):
        """Compute `u` for a tall-and-skinny matrix or `v` for a
        short-and-fat matrix if you have called `fit` with `transform=False`.
        """


class ExactSVD(SVD):
    """Performs an exact Singular Value Decomposition (SVD)
    on a Dask array.

    Inherits from the SVD base class and implements
    Dask's exact SVD algorithm for matrices that are either
    tall-and-skinny or short-and-fat.
    """

    def __init__(self, X):
        super().__init__(X)
        if self.matrix_type == "square":
            msg = (
                "The exact SVD algorithm can only handle tall-and-skinny "
                "or short-and-fat matrices, i.e. the aspect ratio must be "
                ">= 10. Try using the randomized SVD algorithm instead."
            )
            logger.exception(msg)
            raise RuntimeError(msg)

    def fit(self, n_components=-1, transform=False, **kwargs):
        """The default value for `n_components` is -1, meaning
        keep all SVD components. For additional keyword arguments,
        see the documentation for `dask.array.linalg.svd`.
        """
        self.n_components = n_components
        self.X = self.X.persist()
        try:
            logger.info("Fitting exact SVD...")
            u, s, v = da.linalg.svd(self.X, **kwargs)
            if self.n_components != -1:
                u = u[:, : self.n_components]
                s = s[: self.n_components]
                v = v[: self.n_components, :]
            if transform:
                u, s, v = persist(u, s, v)
            elif self.matrix_type == "tall-and-skinny":
                s, v = persist(s, v)
            elif self.matrix_type == "short-and-fat":
                s, u = persist(s, u)
            self.u = u
            self.v = v
            self.s = s.compute()
            logger.info("Finished fitting exact SVD.")
        except Exception as e:
            msg = "Failed fitting exact SVD."
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def transform(self):
        if self.n_components == 0:
            msg = "You have to call `fit` before you can call `transform`."
            logger.exception(msg)
            raise RuntimeError(msg)
        try:
            if self.matrix_type == "tall-and-skinny":
                self.u = self.u.persist()
            if self.matrix_type == "short-and-fat":
                self.v = self.v.persist()
        except Exception as e:
            msg = "Failed transforming input matrix."
            logger.exception(msg)
            raise RuntimeError(msg) from e


class TruncatedSVD(SVD):
    """
    TruncatedSVD performs truncated Singular Value Decomposition (SVD)
    on a tall-and-skinny matrix.

    This class supports both tall-and-skinny and short-and-fat matrices,
    but not square matrices. If the matrix is short-and-fat, it transposes
    the matrix before fitting the truncated SVD. For square matrices,
    consider using the randomized SVD algorithm instead.

    Parameters
    ----------
    X : dask.array.Array
        Input matrix to decompose.

    Attributes
    ----------
    u : dask.array.Array
        Left singular vectors.
    v : dask.array.Array
        Right singular vectors.
    s : numpy.ndarray
        Singular values.

    Methods
    -------
    fit(n_components, **kwargs)
        Compute the truncated SVD of the input matrix, keeping the
        specified number of components. Passes additional keyword
        arguments to the underlying SVD computation (see
        `dask_ml.decomposition.TruncatedSVD` for details).

    Raises
    ------
    RuntimeError
        If input matrix is square/nearly square or if SVD computation fails.
    """

    def __init__(self, X):
        super().__init__(X)
        if self.matrix_type == "square":
            msg = (
                "The truncated SVD algorithm can only handle tall-and-skinny "
                "or short-and-fat matrices. "
                "For square/nearly-square matrices, try using the randomized "
                "SVD algorithm instead."
            )
            logger.exception(msg)
            raise RuntimeError(msg)

    def fit(self, n_components, **kwargs):
        logger.info("Fitting truncated SVD...")
        decomposer = tsvd(n_components=n_components, **kwargs)

        try:
            if self.matrix_type == "tall-and-skinny":
                logger.info("Using truncated SVD for tall-and-skinny matrix.")
                X_transformed = decomposer.fit_transform(self.X)
                s = decomposer.singular_values_
                v_np = decomposer.components_
                u = X_transformed / s  # unscaled

                v = da.from_array(v_np, chunks=v_np.shape)

            elif self.matrix_type == "short-and-fat":
                logger.info(
                    "Using *transposed* truncated SVD for short-and-fat matrix."
                )
                X_transformed_t = decomposer.fit_transform(self.X.T)
                s = decomposer.singular_values_
                v_t_np = decomposer.components_
                u_t = X_transformed_t / s  # unscaled

                u_np = v_t_np.T
                v = u_t.T

                u = da.from_array(u_np, chunks=u_np.shape)

            self.u = u
            self.v = v
            self.s = s  # numpy array

            logger.info("Finished fitting truncated SVD.")

        except Exception as e:
            msg = "Failed fitting truncated SVD."
            logger.exception(msg)
            raise RuntimeError(msg) from e


class RandomizedSVD(SVD):
    """
    RandomizedSVD performs a truncated randomized Singular Value
    Decomposition (SVD) on large matrices of any shape.

    Upon initialization, the input array is rechunked if necessary to
    optimize the computation based on its shape (tall-and-skinny or
    short-and-fat). The fit method computes the truncated SVD using
    Dask's `svd_compressed` function.

    Parameters
    ----------
    X : dask.array.Array
        The input data array to decompose.

    Attributes
    ----------
    u : dask.array.Array
        The left singular vectors of the input matrix.
    s : numpy.ndarray
        The singular values of the input matrix.
    v : dask.array.Array
        The right singular vectors of the input matrix.

    Methods
    -------
    fit(n_components, **kwargs)
        Computes the truncated randomized SVD with the specified
        number of components. Passes additional keyword
        arguments to the underlying SVD computation (see
        `dask.array.linalg.svd_compressed` for details).

    Raises
    ------
    RuntimeError
        If the randomized SVD computation fails.
    """

    def __init__(self, X):
        super().__init__(X)

    def fit(self, n_components, **kwargs):
        self.X = self.X.persist()
        try:
            logger.info("Fitting randomized SVD...")
            u, s, v = da.linalg.svd_compressed(self.X, n_components, **kwargs)
            u, s, v = persist(u, s, v)
            self.u = u[:, :n_components]
            self.v = v[:n_components, :]
            self.s = s[:n_components].compute()
            logger.info("Finished fitting randomized SVD.")
        except Exception as e:
            msg = "Failed fitting randomized SVD."
            logger.exception(msg)
            raise RuntimeError(msg) from e
