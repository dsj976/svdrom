import numpy as np
import pytest
import xarray as xr
from make_test_data import DataGenerator

from svdrom.dmd import OptDMD

generator = DataGenerator()
generator.generate_svd_results(n_components=10)

optdmd = OptDMD()
optdmd_bagging = OptDMD(num_trials=5)


@pytest.mark.parametrize("solver", [optdmd, optdmd_bagging])
def test_basic(solver):
    """Basic test for the OptDMD class."""
    assert hasattr(solver, "modes"), "OptDMD object is missing the 'modes' attribute."
    assert hasattr(solver, "eigs"), "OptDMD object is missing the 'eigs' attribute."
    assert hasattr(
        solver, "amplitudes"
    ), "OptDMD object is missing the 'amplitudes' attribute."
    assert hasattr(
        solver, "modes_std"
    ), "OptDMD object is missing the 'modes_std' attribute."
    assert hasattr(
        solver, "eigs_std"
    ), "OptDMD object is missing the 'eigs_std' attribute."
    assert hasattr(
        solver, "amplitudes_std"
    ), "OptDMD object is missing the 'amplitudes_std' attribute."
    assert hasattr(
        solver, "time_fit"
    ), "OptDMD object is missing the 'time_fit' attribute."


@pytest.mark.parametrize("solver", [optdmd, optdmd_bagging])
def test_fit_basic(solver):
    """Test the fit() method of the OptDMD class."""
    solver.fit(
        generator.u,
        generator.s,
        generator.v,
        varpro_opts_dict={"maxiter": 5},
    )


@pytest.mark.parametrize("solver", [optdmd, optdmd_bagging])
def test_fit_outputs(solver):
    """Test data types and shapes of attributes after
    calling the fit() method."""
    assert isinstance(solver.modes, xr.DataArray), (
        "Expected 'modes' to be of type 'xr.DataArray', "
        f"but got {type(solver.modes)} instead."
    )
    assert isinstance(solver.eigs, np.ndarray), (
        "Expected 'eigs' to be of type 'np.ndarray', "
        f"but got {type(solver.eigs)} instead."
    )
    assert isinstance(solver.amplitudes, np.ndarray), (
        "Expected 'amplitudes' to be of type 'np.ndarray', "
        f"but got {type(solver.amplitudes)} instead."
    )
    assert isinstance(solver.time_fit, np.ndarray), (
        "Expected 'time_fit' to be of type 'np.ndarray', "
        f"but got {type(solver.time_fit)} instead."
    )
    assert solver.modes.shape == generator.u.shape, (
        f"Expected 'modes.shape' to be {generator.u.shape}, "
        f"but got {solver.modes.shape} instead."
    )
    assert solver.eigs.shape == (solver.modes.shape[1],), (
        f"Expected 'eigs.shape' to be {(solver.modes.shape[1],)}, "
        f"but got {solver.eigs.shape} instead."
    )
    assert solver.amplitudes.shape == (solver.modes.shape[1],), (
        f"Expected 'amplitudes.shape' to be {(solver.modes.shape[1],)}, "
        f"but got {solver.amplitudes.shape} instead."
    )
    if solver.num_trials == 0:
        assert solver.modes_std is None, (
            "Expected 'modes_std' to be None, " f"but got {solver.modes_std} instead."
        )
        assert solver.eigs_std is None, (
            "Expected 'eigs_std' to be None, " f"but got {solver.eigs_std} instead."
        )
        assert solver.amplitudes_std is None, (
            "Expected 'amplitudes_std' to be None, "
            f"but got {solver.amplitudes_std} instead."
        )
    else:
        assert isinstance(solver.modes_std, xr.DataArray), (
            "Expected 'modes_std' to be xr.DataArray, "
            f"but got {solver.modes_std} instead."
        )
        assert isinstance(solver.eigs_std, np.ndarray), (
            "Expected 'eigs_std' to be np.ndarray, "
            f"but got {solver.eigs_std} instead."
        )
        assert isinstance(solver.amplitudes_std, np.ndarray), (
            "Expected 'amplitudes_std' to be np.ndarray, "
            f"but got {solver.amplitudes_std} instead."
        )


@pytest.mark.parametrize("forecast_span", ["10 s", 10])
@pytest.mark.parametrize("dt", ["1 s", 10, None])
def test_generate_forecast_time_vector(forecast_span, dt):
    """Test the method to generate the forecast time vector."""
    t_forecast, time_forecast = optdmd._generate_forecast_time_vector(forecast_span, dt)
    assert isinstance(t_forecast, np.ndarray), (
        "Expected 't_forecast' to be of type 'np.ndarray', "
        f"but got {type(t_forecast)} instead."
    )
    assert isinstance(time_forecast, np.ndarray), (
        "Expected 'time_forecast' to be of type 'np.ndarray', "
        f"but got {type(time_forecast)} instead."
    )
    assert np.unique(np.diff(t_forecast)) == np.timedelta64(1, "s"), (
        "Expected the time difference between consecutive elements in "
        "'t_forecast' to be one second, but got "
        f"{np.unique(np.diff(t_forecast))} instead."
    )
    assert np.unique(np.diff(time_forecast)) == np.timedelta64(1, "s"), (
        "Expected the time difference between consecutive elements in "
        "'time_forecast' to be one second, but got "
        f"{np.unique(np.diff(time_forecast))} instead."
    )
    assert len(t_forecast) == 10, (
        "Expected the length of 't_forecast' to be 10, but got "
        f"{len(t_forecast)} instead."
    )
    assert len(time_forecast) == 10, (
        "Expected the length of 'time_forecast' to be 10, but got "
        f"{len(time_forecast)} instead."
    )
    assert t_forecast[0] == optdmd._t_fit[-1] + np.timedelta64(1, "s"), (
        "Expected 't_forecast[0]' to be one second ahead"
        f"of t_fit[-1], but got {t_forecast[0] - optdmd._t_fit[-1]} instead."
    )
    assert time_forecast[0] == optdmd.time_fit[-1] + np.timedelta64(1, "s"), (
        "Expected 'time_forecast[0]' to be one second ahead of time_fit[-1], "
        f"but got {time_forecast[0] - optdmd.time_fit[-1]} instead."
    )


@pytest.mark.parametrize("solver", [optdmd, optdmd_bagging])
def test_forecast(solver):
    """Test for the forecast() method."""
    if solver.num_trials == 0:
        forecast = solver.forecast(forecast_span="10 s", dt="1 s")
        assert isinstance(forecast, xr.DataArray), (
            "Expected 'forecast' to be of type 'xr.DataArray', "
            f"but got {type(forecast)} instead."
        )
    else:
        forecast, forecast_var = solver.forecast(forecast_span="10 s", dt="1 s")
        assert isinstance(forecast, xr.DataArray), (
            "Expected 'forecast' to be of type 'xr.DataArray', "
            f"but got {type(forecast)} instead."
        )
        assert isinstance(forecast_var, xr.DataArray), (
            "Expected 'forecast_var' to be of type 'xr.DataArray', "
            f"but got {type(forecast)} instead."
        )
