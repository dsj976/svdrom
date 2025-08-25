import numpy as np
import xarray as xr
from pydmd import BOPDMD

from svdrom.logger import setup_logger

logger = setup_logger("DMD", "dmd.log")


class OptDMD:
    def __init__(
        self,
        n_modes: int = -1,
        time_dimension: str = "time",
    ) -> None:
        if n_modes != -1 and n_modes < 1:
            msg = "'n_modes' must be a positive integer or -1."
            logger.exception(msg)
            raise ValueError(msg)
        self._n_modes = n_modes
        self._time_dimension = time_dimension
        self._eigs: np.ndarray | None = None
        self._amplitudes: np.ndarray | None = None
        self._modes: xr.DataArray | None = None

    @property
    def n_modes(self) -> int:
        """Number of DMD modes (read-only)."""
        return self._n_modes

    @property
    def time_dimension(self) -> str:
        """Name of the time dimension (read-only)."""
        return self._time_dimension

    @property
    def eigs(self) -> np.ndarray | None:
        """The DMD eigenvalues (read-only)."""
        return self._eigs

    @property
    def modes(self) -> xr.DataArray | None:
        """The DMD modes (read-only)."""
        return self._modes

    @property
    def amplitudes(self) -> np.ndarray | None:
        """The DMD amplitudes (read-only)."""
        return self._amplitudes

    def _check_svd_inputs(self, u: xr.DataArray, s: np.ndarray, v: xr.DataArray):
        """Check that the passed SVD results are valid."""
        if not isinstance(u.data, np.ndarray):
            msg = "The left singular vectors have not been computed."
            logger.exception(msg)
            raise ValueError(msg)
        if not isinstance(v.data, np.ndarray):
            msg = "The right singular vectors have not been computed."
            logger.exception(msg)
            raise ValueError(msg)
        if len(u.components) != len(v.components) or len(u.components) != len(s):
            msg = "'u', 's' and 'v' must have the same number of components."
            logger.exception(msg)
            raise ValueError(msg)
        if self._time_dimension not in v.dims:
            msg = (
                f"Specified time dimension '{self._time_dimension}' not "
                "a dimension of the right singular vectors 'v'."
            )
            logger.exception(msg)
            raise ValueError(msg)

        def is_sorted(x):
            return np.all(x[:-1] <= x[1:])

        if not is_sorted(v[self._time_dimension].values):
            msg = (
                f"Time dimension '{self._time_dimension}' is not "
                "sorted in the right singular vectors 'v'."
            )
            logger.exception(msg)
            raise ValueError(msg)

    def _generate_fit_time_vector(self, v: xr.DataArray) -> np.ndarray:
        """Given the right singular vectors containing the temporal
        information, generate the time vector for the DMD fit.
        """

    def fit(self, u: xr.DataArray, s: np.ndarray, v: xr.DataArray):
        self._check_svd_inputs(u, s, v)
        if self._n_modes > len(s):
            msg = (
                "The requested number of DMD modes exceeds the number "
                "of available SVD components."
            )
            logger.exception(msg)
            raise ValueError(msg)
        if self._n_modes == -1:
            self._n_modes = len(s)

        bopdmd = BOPDMD(
            svd_rank=self._n_modes, use_proj=True, proj_basis=u[:, : self._n_modes]
        )
