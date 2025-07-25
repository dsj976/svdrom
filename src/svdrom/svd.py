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
        self._n_components = 0
        self._u = da.empty_like(0)
        self._s = np.empty_like(0)
        self._v = da.empty_like(0)
        if X.ndim != 2:
            msg = "The input array must be two-dimensional."
            logger.exception(msg)
            raise ValueError(msg)
        self._X = X
        self._matrix_type = ""
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
        n_rows, n_cols = self._X.shape
        if (n_rows // n_cols) >= aspect_ratio:
            self._matrix_type = "tall-and-skinny"
        elif (n_cols // n_rows) >= aspect_ratio:
            self._matrix_type = "short-and-fat"
        else:
            self._matrix_type = "square"

    def _rechunk_array(self):
        """Rechunks the input array to ensure optimal chunk sizes
        for SVD computation based on matrix type.
        """
        msg = (
            "Will need to rechunk the array before fitting the SVD. "
            "This will add some overhead."
        )
        if (
            self._matrix_type == "tall-and-skinny"
            and self._X.shape[1] != self._X.chunksize[1]
        ):
            logger.info(msg)
            self._X = self._X.rechunk({0: "auto", 1: -1})
        if (
            self._matrix_type == "short-and-fat"
            and self._X.shape[0] != self._X.chunksize[0]
        ):
            logger.info(msg)
            self._X = self._X.rechunk({0: -1, 1: "auto"})

    @property
    def n_components(self) -> int:
        """Number of SVD components (read-only)."""
        return self._n_components

    @property
    def s(self) -> np.ndarray:
        """Singular values (read-only)."""
        return self._s

    @property
    def u(self) -> da.Array:
        """Left singular vectors (read-only)."""
        return self._u

    @property
    def v(self) -> da.Array:
        """Right singular vectors (read-only)."""
        return self._v

    @property
    def X(self) -> da.Array:
        """Input matrix (read-only)."""
        return self._X

    @property
    def matrix_type(self) -> str:
        """Matrix type, based on aspect radio (read-only)."""
        return self._matrix_type

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
        if self._matrix_type == "square":
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
        if n_components != -1 and (
            not isinstance(n_components, int) or n_components <= 0
        ):
            msg = "n_components must be -1 or a positive integer."
            logger.exception(msg)
            raise ValueError(msg)
        self._n_components = n_components
        try:
            logger.info("Fitting exact SVD...")
            u, s, v = da.linalg.svd(self._X, **kwargs)
            if self._n_components != -1:
                u = u[:, : self._n_components]
                s = s[: self._n_components]
                v = v[: self._n_components, :]
            if transform:
                u, s, v = persist(u, s, v)
            elif self._matrix_type == "tall-and-skinny":
                s, v = persist(s, v)
            elif self._matrix_type == "short-and-fat":
                s, u = persist(s, u)
            self._u = u
            self._v = v
            self._s = s.compute()
            logger.info("Finished fitting exact SVD.")
        except Exception as e:
            msg = "Failed fitting exact SVD."
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def transform(self):
        if self._n_components == 0:
            msg = "You have to call `fit` before you can call `transform`."
            logger.exception(msg)
            raise RuntimeError(msg)
        try:
            if self._matrix_type == "tall-and-skinny":
                self._u = self._u.persist()
            if self._matrix_type == "short-and-fat":
                self._v = self._v.persist()
        except Exception as e:
            msg = "Failed transforming input matrix."
            logger.exception(msg)
            raise RuntimeError(msg) from e


class TruncatedSVD(SVD):
    """Performs a truncated Singular Value Decomposition (SVD)
    on a Dask array.

    Inherits from the SVD base class.
    Supports both tall-and-skinny and short-and-fat matrices,
    but not square matrices. If the matrix is short-and-fat, it transposes
    the matrix into a tall-and-skinny one before fitting the truncated SVD.
    """

    def __init__(self, X):
        super().__init__(X)
        self._decomposer = None
        if self._matrix_type == "square":
            msg = (
                "The truncated SVD algorithm can only handle tall-and-skinny "
                "or short-and-fat matrices. "
                "For square/nearly-square matrices, try using the randomized "
                "SVD algorithm instead."
            )
            logger.exception(msg)
            raise RuntimeError(msg)

    def fit(self, n_components, transform=False, **kwargs):
        """For additional keyword arguments, see the documentation
        for `dask_ml.decomposition.TruncatedSVD`.
        """
        if not isinstance(n_components, int) or n_components <= 0:
            msg = "n_components must be a positive integer."
            logger.exception(msg)
            raise ValueError(msg)
        self._n_components = n_components

        logger.info("Fitting truncated SVD...")
        self._decomposer = tsvd(
            n_components=self._n_components, algorithm="tsqr", **kwargs
        )

        try:
            if self._matrix_type == "tall-and-skinny":
                logger.info("Using truncated SVD for tall-and-skinny matrix.")
                if transform:
                    X_transformed = self._decomposer.fit_transform(self._X)
                    self._u = (
                        X_transformed / self._decomposer.singular_values_
                    )  # unscaled
                else:
                    self._decomposer.fit(self._X)
                self._s = self._decomposer.singular_values_  # numpy array
                v_np = self._decomposer.components_
                self._v = da.from_array(v_np, chunks=v_np.shape)

            elif self._matrix_type == "short-and-fat":
                logger.info(
                    "Using *transposed* truncated SVD for short-and-fat matrix."
                )
                if transform:
                    X_transformed_t = self._decomposer.fit_transform(self._X.T)
                    u_t = (
                        X_transformed_t / self._decomposer.singular_values_
                    )  # unscaled
                    self._v = u_t.T
                else:
                    self._decomposer.fit(self._X.T)
                self._s = self._decomposer.singular_values_  # numpy array
                v_t_np = self._decomposer.components_
                u_np = v_t_np.T
                self._u = da.from_array(u_np, chunks=u_np.shape)

            logger.info("Finished fitting truncated SVD.")

        except Exception as e:
            msg = "Failed fitting truncated SVD."
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def transform(self):
        if self._decomposer is None:
            msg = "You have to call `fit` before you can call `transform`."
            logger.exception(msg)
            raise RuntimeError(msg)
        try:
            if self._matrix_type == "tall-and-skinny":
                X_transformed = self._decomposer.transform(self._X)
                self._u = X_transformed / self._decomposer.singular_values_  # unscaled
            if self._matrix_type == "short-and-fat":
                X_transformed_t = self._decomposer.transform(self._X.T)
                u_t = X_transformed_t / self._decomposer.singular_values_  # unscaled
                self._v = u_t.T
        except Exception as e:
            msg = "Failed transforming input matrix."
            logger.exception(msg)
            raise RuntimeError(msg) from e


class RandomizedSVD(SVD):
    """Performs a truncated randomized Singular Value
    Decomposition (SVD) on large matrices of any shape.

    Inherits from the SVD base class.
    Supports tall-and-skinny, short-and-fat, and square/nearly-square
    matrices. The SVD computation is performed with
    `dask.array.linalg.svd_compressed`.
    """

    def __init__(self, X):
        super().__init__(X)

    def fit(self, n_components, transform=False, **kwargs):
        """Note that if the matrix is square/nearly-square, if `transform=False`
        only the singular values and right singular vectors will be computed
        (i.e. same behavior as a tall-and-skinny matrix).

        For additional keyword arguments, see the documentation
        for `dask.array.linalg.svd_compressed`.
        """
        if not isinstance(n_components, int) or n_components <= 0:
            msg = "n_components must be a positive integer."
            logger.exception(msg)
            raise ValueError(msg)
        self._n_components = n_components
        try:
            logger.info("Fitting randomized SVD...")
            u, s, v = da.linalg.svd_compressed(self._X, n_components, **kwargs)
            if transform:
                u, s, v = persist(u, s, v)
            elif (
                self._matrix_type == "tall-and-skinny-matrix"
                or self._matrix_type == "square"
            ):
                s, v = persist(s, v)
            elif self._matrix_type == "short-and-fat":
                s, u = persist(s, u)
            self._u = u
            self._v = v
            self._s = s.compute()
            logger.info("Finished fitting randomized SVD.")
        except Exception as e:
            msg = "Failed fitting randomized SVD."
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def transform(self):
        if self._n_components == 0:
            msg = "You have to call `fit` before you can call `transform`."
            logger.exception(msg)
            raise RuntimeError(msg)
        try:
            if self._matrix_type == "tall-and-skinny" or self._matrix_type == "square":
                self._u = self._u.persist()
            if self._matrix_type == "short-and-fat":
                self._v = self._v.persist()
        except Exception as e:
            msg = "Failed transforming input matrix."
            logger.exception(msg)
            raise RuntimeError(msg) from e
