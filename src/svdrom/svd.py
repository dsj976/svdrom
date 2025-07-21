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
        rechunk: bool = True,
        aspect_ratio: int = 10,
    ):
        self._n_components = n_components
        self._algorithm = algorithm
        self._rechunk = rechunk
        self._aspect_ratio = aspect_ratio
        self._u = xr.DataArray | None
        self._s = np.ndarray | None
        self._v = xr.DataArray | None
        self._matrix_type: str | None = None

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
        compute_u: bool = True,
        compute_v: bool = True,
        compute_var_ratio: bool = False,
    ):
        pass

    def transform(self, X: xr.DataArray):
        pass
