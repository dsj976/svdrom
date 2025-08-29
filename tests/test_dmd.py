import numpy as np
import xarray as xr
from make_test_data import DataGenerator

from svdrom.dmd import OptDMD

generator = DataGenerator()
generator.generate_svd_results(n_components=10)

optdmd = OptDMD()


def test_fit_basic():
    """Basic test for the fit() method of the OptDMD class."""
    optdmd.fit(
        generator.u,
        generator.s,
        generator.v,
        varpro_opts_dict={"maxiter": 3},
    )
    assert hasattr(optdmd, "modes"), "OptDMD object is missing the 'modes' attribute."
    assert hasattr(optdmd, "eigs"), "OptDMD object is missing the 'eigs' attribute."
    assert hasattr(
        optdmd, "amplitudes"
    ), "OptDMD object is missing the 'amplitudes' attribute."
    assert hasattr(
        optdmd, "modes_std"
    ), "OptDMD object is missing the 'modes_std' attribute."
    assert hasattr(
        optdmd, "eigs_std"
    ), "OptDMD object is missing the 'eigs_std' attribute."
    assert hasattr(
        optdmd, "amplitudes_std"
    ), "OptDMD object is missing the 'amplitudes_std' attribute."
    assert hasattr(optdmd, "t_fit"), "OptDMD object is missing the 't_fit' attribute."


def test_fit_outputs():
    """Test data types and shapes of attributes after
    calling the fit() method."""
    assert isinstance(optdmd.modes, xr.DataArray), (
        "Expected 'optdmd.modes' to be of type 'xr.DataArray', "
        f"but got {type(optdmd.modes)} instead."
    )
    assert isinstance(optdmd.eigs, np.ndarray), (
        "Expected 'optdmd.eigs' to be of type 'np.ndarray', "
        f"but got {type(optdmd.eigs)} instead."
    )
    assert isinstance(optdmd.amplitudes, np.ndarray), (
        "Expected 'optdmd.amplitudes' to be of type 'np.ndarray', "
        f"but got {type(optdmd.amplitudes)} instead."
    )
    assert optdmd.modes_std is None, (
        "Expected 'optdmd.modes_std' to be None, "
        f"but got {optdmd.modes_std} instead."
    )
    assert optdmd.eigs_std is None, (
        "Expected 'optdmd.eigs_std' to be None, " f"but got {optdmd.eigs_std} instead."
    )
    assert optdmd.amplitudes_std is None, (
        "Expected 'optdmd.amplitudes_std' to be None, "
        f"but got {optdmd.amplitudes_std} instead."
    )
    assert optdmd.modes.shape == generator.u.shape, (
        f"Expected 'optdmd.modes.shape' to be {generator.u.shape}, "
        f"but got {optdmd.modes.shape} instead."
    )
    assert optdmd.eigs.shape == (optdmd.modes.shape[1],), (
        f"Expected 'optdmd.eigs.shape' to be {(optdmd.modes.shape[1],)}, "
        f"but got {optdmd.eigs.shape} instead."
    )
    assert optdmd.amplitudes.shape == (optdmd.modes.shape[1],), (
        f"Expected 'optdmd.amplitudes.shape' to be {(optdmd.modes.shape[1],)}, "
        f"but got {optdmd.amplitudes.shape} instead."
    )
