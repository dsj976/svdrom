import dask
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

    @property
    def algorithm(self):
        """SVD algorithm to use (read-only)."""
        return self._algorithm

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

    def _rechunk_array(self, X: da.Array) -> da.Array:
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

    def _check_array(self, X: xr.DataArray):
        if X.ndim != 2:
            msg = (
                "The input array must be 2-dimensional. "
                f"Got a {X.ndim}-dimensional array."
            )
            logger.exception(msg)
            raise ValueError(msg)
        if self._n_components >= X.shape[1]:
            msg = (
                "n_components must be less than n_features. "
                f"Got n_components: {self.n_components}, n_features: {X.shape[1]}."
            )
            logger.exception(msg)
            raise ValueError(msg)
        if not isinstance(X.data, da.Array):
            msg = (
                f"The {self.__class__.__name__} class only supports Dask-backed "
                f"Xarray DataArrays. Got {type(X.data)} instead."
            )
            logger.exception(msg)
            raise TypeError(msg)

    def _to_dataarray(
        self, singular_vectors: np.ndarray, X: xr.DataArray
    ) -> xr.DataArray:
        """Transform the singular vectors into a Xarray DataArray following
        the dimensions and coordinates of the DataArray on which SVD was
        performed. The function automatically identifies whether the input
        singular vectors are left or right singular vectors.
        """
        if singular_vectors.shape[0] == X.shape[0]:
            # this corresponds to `u`: replace second dimension (e.g. 'time')
            old_dims = list(X.dims)
            new_dims = [old_dims[0], "components"]
            coords = {k: v for k, v in X.coords.items() if k != old_dims[1]}
            coords["components"] = np.arange(singular_vectors.shape[1])
            name = "svd_u"
        elif singular_vectors.shape[1] == X.shape[1]:
            # this corresponds to `v`: replace first dimension (e.g. 'samples')
            old_dims = list(X.dims)
            new_dims = ["components", old_dims[1]]
            coords = {k: v for k, v in X.coords.items() if k != old_dims[0]}
            coords["components"] = np.arange(singular_vectors.shape[0])
            name = "svd_v"
        else:
            msg = (
                "Cannot transform singular vectors into Xarray DataArray. "
                "Shape of singular_vectors does not match X."
            )
            logger.exception(msg)
            raise ValueError(msg)
        return xr.DataArray(singular_vectors, dims=new_dims, coords=coords, name=name)

    def fit(
        self,
        X: xr.DataArray,
        **kwargs,
    ):
        if self._algorithm not in ["tsqr", "randomized"]:
            msg = (
                f"Unsupported algorithm: {self._algorithm}. "
                "Supported algorithms are 'tsqr' and 'randomized'."
            )
            logger.exception(msg)
            raise ValueError(msg)

        self._check_array(X)
        X_da = X.data
        X_da = self._rechunk_array(X_da) if self._rechunk else X_da

        if self._algorithm == "tsqr":
            msg = "Will use TSQR algorithm."
            logger.info(msg)
            u, s, v = da.linalg.svd(X_da)  # employs tsqr internally
            u = u[:, : self._n_components]
            s = s[: self._n_components]
            v = v[: self._n_components]
        else:
            msg = "Will use randomized algorithm."
            logger.info(msg)
            u, s, v = da.linalg.svd_compressed(X_da, self._n_components, **kwargs)

        X_da_transformed = u * s
        explained_var = X_da_transformed.var(axis=0)
        full_var = X_da.var(axis=0).sum()
        explained_var_ratio = explained_var / full_var

        results = []
        results.append(s)

        if self._compute_u:
            results.append(u)
        if self._compute_v:
            results.append(v)
        if self._compute_var_ratio:
            results.append(explained_var_ratio)

        # compute all Dask collections at once
        msg = "Computing SVD results..."
        logger.info(msg)
        computed = dask.compute(*results)
        msg = "Done."
        logger.info(msg)

        s = computed[0]
        idx = 1
        if self._compute_u:
            u = computed[idx]
            idx += 1
        if self._compute_v:
            v = computed[idx]
            idx += 1
        if self._compute_var_ratio:
            explained_var_ratio = computed[idx]

    def transform(self, X: xr.DataArray):
        pass
