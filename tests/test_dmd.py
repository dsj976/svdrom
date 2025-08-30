import numpy as np
import pytest
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
    assert hasattr(
        optdmd, "time_fit"
    ), "OptDMD object is missing the 'time_fit' attribute."
    assert hasattr(
        optdmd, "time_forecast"
    ), "OptDMD object is missing the 'time_forecast' attribute."


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
    assert isinstance(optdmd.time_fit, np.ndarray), (
        "Expected 'optdmd.time_fit' to be of type 'np.ndarray', "
        f"but got {type(optdmd.time_fit)} instead."
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


@pytest.mark.parametrize("forecast_span", ["10 s", 10])
@pytest.mark.parametrize("dt", ["1 s", 10, None])
def test_generate_forecast_time_vector(forecast_span, dt):
    """Test the method to generate the forecast time vector."""
    t_forecast = optdmd._generate_forecast_time_vector(forecast_span, dt)
    assert isinstance(t_forecast, np.ndarray), (
        "Expected 't_forecast' to be of type 'np.ndarray', "
        f"but got {type(t_forecast)} instead."
    )
    assert isinstance(optdmd.time_forecast, np.ndarray), (
        "Expected 'time_forecast' to be of type 'np.ndarray', "
        f"but got {type(optdmd.time_forecast)} instead."
    )
    assert np.unique(np.diff(t_forecast)) == np.timedelta64(1, "s"), (
        "Expected the time difference between consecutive elements in "
        "'t_forecast' to be one second, but got "
        f"{np.unique(np.diff(t_forecast))} instead."
    )
    assert np.unique(np.diff(optdmd.time_forecast)) == np.timedelta64(1, "s"), (
        "Expected the time difference between consecutive elements in "
        "'time_forecast' to be one second, but got "
        f"{np.unique(np.diff(optdmd.time_forecast))} instead."
    )
    assert len(t_forecast) == 10, (
        "Expected the length of 't_forecast' to be 10, but got "
        f"{len(t_forecast)} instead."
    )
    assert len(optdmd.time_forecast) == 10, (
        "Expected the length of 'time_forecast' to be 10, but got "
        f"{len(optdmd.time_forecast)} instead."
    )
    assert t_forecast[0] == optdmd._t_fit[-1] + np.timedelta64(1, "s"), (
        "Expected 't_forecast[0]' to be one second ahead"
        f"of t_fit[-1], but got {t_forecast[0] - optdmd._t_fit[-1]} instead."
    )
    assert optdmd.time_forecast[0] == optdmd.time_fit[-1] + np.timedelta64(1, "s"), (
        "Expected 'time_forecast[0]' to be one second ahead of time_fit[-1], "
        f"but got {optdmd.time_forecast[0] - optdmd.time_fit[-1]} instead."
    )
