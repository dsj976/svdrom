import numpy as np

from svdrom.logger import setup_logger
from svdrom.svd import TruncatedSVD

logger = setup_logger("DMD", "dmd.log")


class OptDMD:
    def __init__(
        self,
        tsvd: TruncatedSVD,
        n_modes: int = -1,
    ) -> None:
        self._check_tsvd(tsvd)
        self._tsvd = tsvd
        if n_modes == -1:
            self._n_modes = tsvd.n_components
        elif n_modes < 1:
            msg = "'n_modes' must be a positive integer or -1."
            raise ValueError(msg)
        elif n_modes > tsvd.n_components:
            msg = (
                "'n_modes' cannot be greater than the number of "
                "available SVD components."
            )
        else:
            self._n_modes = n_modes

    @property
    def n_modes(self) -> int:
        """Number of DMD modes (read-only)."""
        return self._n_modes

    @property
    def tsvd(self) -> TruncatedSVD:
        """The TruncatedSVD instance used to fit
        the DMD model (read-only)."""
        return self._tsvd

    @staticmethod
    def _check_tsvd(tsvd: TruncatedSVD):
        """Check that the TruncatedSVD instance is valid."""
        if tsvd.u is None:
            msg = "The left singular vectors have not been computed."
            raise ValueError(msg)
        if not isinstance(tsvd.u.data, np.ndarray):
            msg = "The left singular vectors have not been computed."
            raise ValueError(msg)
        if tsvd.v is None:
            msg = "The right singular vectors have not been computed."
            raise ValueError(msg)
        if not isinstance(tsvd.v.data, np.ndarray):
            msg = "The right singular vectors have not been computed."
            raise ValueError(msg)
