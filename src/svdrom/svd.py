import dask.array as da
import numpy as np
import xarray as xr

from svdrom.logger import setup_logger

logger = setup_logger("SVD", "svd.log")


class TruncatedSVD:
    def __init__(
        self,
        n_components: int,
        algorithm: str = "tsqr",
        compute_u: bool = True,
        compute_v: bool = True,
        compute_var_ratio: bool = False,
        rechunk: bool = True,
        aspect_ratio: int = 10,
    ):
        self._n_components = n_components
        self._algorithm = algorithm
        self._compute_u = compute_u
        self._compute_v = compute_v
        self._compute_var_ratio = compute_var_ratio
        self._rechunk = rechunk
        self._aspect_ratio = aspect_ratio
        self._u: xr.DataArray | None = None
        self._s: np.ndarray | None = None
        self._v: xr.DataArray | None = None
        self._matrix_type: str | None = None

    @property
    def n_components(self) -> int:
        """Number of SVD components (read-only)."""
        return self._n_components

    @property
    def s(self):
        """Singular values (read-only)."""
        return self._s

    @property
    def u(self):
        """Left singular vectors (read-only)."""
        return self._u

    @property
    def v(self):
        """Right singular vectors (read-only)."""
        return self._v

    @property
    def matrix_type(self):
        """Matrix type, based on aspect radio (read-only)."""
        return self._matrix_type

    @property
    def compute_u(self):
        """Whether to compute left singular vectors (read-only)."""
        return self._compute_u

    @property
    def compute_v(self):
        """Whether to compute right singular vectors (read-only)."""
        return self._compute_v

    @property
    def compute_var_ratio(self):
        """Whether to compute the ratio of explained variance (read-only)."""
        return self._compute_var_ratio

    @property
    def aspect_ratio(self):
        """Aspect ratio used to determine matrix type (read-only)."""
        return self._aspect_ratio

    @property
    def rechunk(self):
        """Whether to rechunk the input array before fitting (read-only)."""
        return self._rechunk

    def _check_matrix_type(self, X: da.Array):
        """Checks if input matrix is tall-and-skinny,
        short-and-fat or square/nearly-square, based on
        the specified aspect ratio.
        """
        n_rows, n_cols = X.shape
        if (n_rows // n_cols) >= self._aspect_ratio:
            self._matrix_type = "tall-and-skinny"
        elif (n_cols // n_rows) >= self._aspect_ratio:
            self._matrix_type = "short-and-fat"
        else:
            self._matrix_type = "square"

    def _rechunk_array(self, X: da.Array):
        """Rechunks the input array to ensure optimal chunk sizes
        for SVD computation based on matrix type.
        """
        nb = X.numblocks
        msg = (
            "Will rechunk the array before fitting the SVD. "
            "This will add some overhead."
        )
        if self._matrix_type == "tall-and-skinny" and X.shape[1] != nb[1]:
            logger.info(msg)
            return X.rechunk({0: "auto", 1: -1})
        if self._matrix_type == "short-and-fat" and X.shape[0] != nb[0]:
            logger.info(msg)
            return X.rechunk({0: -1, 1: "auto"})
        return X

    def fit(
        self,
        X: xr.DataArray,
    ):
        pass

    def transform(self, X: xr.DataArray):
        pass
