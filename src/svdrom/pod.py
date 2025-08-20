# src/svdrom/pod.py

import numpy as np
import xarray as xr

from svdrom.logger import setup_logger
from svdrom.svd import TruncatedSVD

logger = setup_logger("POD", "pod.log")

# LSV - modes
# RSV - Time coefficients  # the POD modes
# return SV squared / energy
# flag for remove mean along temporal dimension in fit or class init?
# important to remove time average

# The shape of the array matters when you feed it to SVD. 
# Whether you have spatial dim along rows or columns

# if you call modes, you're essentially returning the RSV
# time coefficients you return LSV
# The algo should figure out which is the spatioal and temporal dims!


# Inherit TruncatedSVD maybe!
class POD:
    """
    Proper Orthogonal Decomposition (POD)

    Uses SVD to compute the POD modes (spatial modes),
    temporal coefficients, and modal energies of a given dataset.

    Parameters
    ----------
    n_modes : int
        The number of POD modes to compute and keep. This corresponds to
        the number of components in the truncated SVD.
    algorithm : str, {'tsqr', 'randomized'}, (default 'tsqr')
        SVD algorithm to use via the `TruncatedSVD` class.
        See `svdrom.svd.TruncatedSVD` for more details.
    **svd_kwargs : dict
        Additional keyword arguments to be passed to the `TruncatedSVD`
        constructor (e.g., `compute_u`, `compute_v`, `rechunk`).
    """

    def __init__(self, n_modes: int, algorithm: str = "tsqr"):
        self._n_modes = n_modes
        self._algorithm = algorithm
        self._svd_solver = TruncatedSVD(n_components=n_modes, algorithm=algorithm)
        self._modes: xr.DataArray | None = None
        self._coeffs: xr.DataArray | None = None
        self._energy: np.ndarray | None = None
        self._s: np.ndarray | None = None
        self._mean_field: xr.DataArray | None = None
        self._n_snapshots: int | None = None

    @property
    def n_modes(self) -> int:
        return self._n_modes

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def modes(self) -> xr.DataArray | None:
        return self._modes

    @property
    def coeffs(self) -> xr.DataArray | None:
        return self._coeffs

    @property
    def energy(self) -> np.ndarray | None:
        return self._energy

    @property
    def s(self) -> np.ndarray | None:
        return self._s

    @property
    def mean_field(self) -> xr.DataArray | None:
        return self._mean_field

    @property
    def n_snapshots(self) -> int | None:
        return self._n_snapshots

    def fit(self, X: xr.DataArray, dim: str, **svd_kwargs) -> None:
        """
        X : xr.DataArray, shape (n_samples, n_features)
            The input data.
        dim : str
            The name of the dimension in `X` that represents the snapshots.
        """
        if dim not in X.dims:
            msg = (
                f"Dimension '{dim}' not present in the input array with dims {X.dims}."
            )
            logger.exception(msg)
            raise ValueError(msg)

        self._n_snapshots = X.sizes[dim]
        logger.info(f"Computing mean field along dimension '{dim}'...")
        # Remove mean is turned on by default - user warning
        self._mean_field = X.mean(
            dim=dim
        ).compute()  # We could just pass mean; could use StandardScaler instead
        X_fluc = X - self._mean_field

        D = X_fluc / np.sqrt(self._n_snapshots)
        logger.info(f"Calling TruncatedSVD with {self.n_modes} modes...")
        self._svd_solver.fit(D, **svd_kwargs)

        assert self._svd_solver.s is not None

        self._s = self._svd_solver.s
        self._energy = self._s**2

        assert self._svd_solver.u is not None
        assert self._svd_solver.v is not None

        u = self._svd_solver.u
        v = self._svd_solver.v

        self._modes = v.T.rename({"components": "modes"})
        self._coeffs = (u * self._s).rename({"components": "modes"})

    def transform(self, X: xr.DataArray) -> xr.DataArray:
        """Project new data onto the computed POD basis."""
        if self.modes is None or self.mean_field is None or self.n_snapshots is None:
            msg = "The fit() method must be called before calling transform."
            logger.exception(msg)
            raise RuntimeError(msg)

        X_fluc = X - self.mean_field
        coeffs_data = (X_fluc.data @ self.modes.data) / np.sqrt(self.n_snapshots)
        coeffs_computed = coeffs_data.compute()

        time_dim_name = X.dims[0]
        new_coords = {
            time_dim_name: X.coords[time_dim_name],
            "modes": self.modes.coords["modes"],
        }

        return xr.DataArray(
            coeffs_computed,
            dims=[time_dim_name, "modes"],
            coords=new_coords,
            name="transformed_coeffs",
        )

    def reconstruct(self, n_modes: int | None = None) -> xr.DataArray:
        """Reconstruct the full data field using a specified number of modes."""
        if (
            self.coeffs is None
            or self.mean_field is None
            or self.modes is None
            or self.n_snapshots is None
        ):
            msg = "The fit() method must be called before reconstruction."
            logger.exception(msg)
            raise RuntimeError(msg)

        if n_modes is None:
            n_modes = self._n_modes
        elif n_modes > self._n_modes:
            logger.warning(
                f"Requested {n_modes} modes, but only {self._n_modes} are "
                f"available. Using all {self._n_modes} modes."
            )
            n_modes = self._n_modes

        logger.info(f"Reconstructing field using the first {n_modes} modes...")
        coeffs_subset = self.coeffs.sel(modes=slice(0, n_modes))

        X_fluc_recon_data = (coeffs_subset.data @ self.modes.data.T) * np.sqrt(
            self.n_snapshots
        )
        time_dim, _ = coeffs_subset.dims
        space_dim, _ = self.modes.dims
        recon_coords = {
            time_dim: coeffs_subset.coords[time_dim],
            space_dim: self.modes.coords[space_dim],
        }
        X_fluc_recon = xr.DataArray(
            X_fluc_recon_data,
            dims=[time_dim, space_dim],
            coords=recon_coords,
        )

        X_recon_data = X_fluc_recon.data + self.mean_field.data
        X_recon_computed = X_recon_data
        logger.info("Reconstruction complete.")

        return xr.DataArray(
            X_recon_computed,
            dims=X_fluc_recon.dims,
            coords=X_fluc_recon.coords,
            name="reconstructed_field",
        )
