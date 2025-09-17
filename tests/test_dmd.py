import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from make_test_data import DataGenerator

from svdrom.dmd import OptDMD

generator = DataGenerator()
generator.generate_svd_results(n_components=10)

optdmd = OptDMD()
optdmd_bagging = OptDMD(num_trials=5)

# set the dask scheduler to single-threaded
dask.config.set(scheduler="single-threaded")


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
    np.testing.assert_equal(
        solver.time_fit,
        generator.v.time.values,
        strict=True,
        err_msg="Expected 'time_fit' vector to equal 'v.time.values'.",
    )
    assert isinstance(solver._t_fit, np.ndarray), (
        "Expected 't_fit' to be of type 'np.ndarray', "
        f"but got {type(solver._t_fit)} instead."
    )
    assert solver._t_fit.dtype.name == f"timedelta64[{solver._time_units}]", (
        f"Expected 't_fit' vector to have data type timedelta64[{solver._time_units}], "
        f"but got {solver._t_fit.dtype.name}."
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
def test_predict(solver):
    """Test for the private predict() method, ensuring it returns the
    same output as BOPDMD.forecast() from PyDMD.
    """
    t, _ = solver._generate_forecast_time_vector(
        forecast_span="10 s",
        dt="1 s",
    )
    t = t.astype("float64")
    if solver.num_trials == 0:
        # without bagging

        # no Dask
        forecast_np = solver._predict(t, use_dask=False)
        forecast_np_pydmd = solver.solver.forecast(t)
        assert isinstance(forecast_np, np.ndarray), (
            "Expected the forecast to be a np.ndarray, "
            f"but got {type(forecast_np)} instead."
        )
        np.testing.assert_allclose(
            forecast_np.real,
            forecast_np_pydmd.real,
            err_msg="Expected the forecast to match the output of PyDMD.",
        )

        # with Dask
        forecast_da = solver._predict(t, use_dask=True)
        assert isinstance(forecast_da, da.Array), (
            "Expected the forecast to be a da.Array, "
            f"but got {type(forecast_da)} instead."
        )
        forecast_np = forecast_da.compute()
        np.testing.assert_allclose(
            forecast_np.real,
            forecast_np_pydmd.real,
            err_msg="Expected the forecast to match the output of PyDMD.",
        )
    else:
        # with bagging

        # no Dask
        np.random.seed(42)
        forecast_np, forecast_var_np = solver._predict(t, use_dask=False)
        np.random.seed(42)
        forecast_np_pydmd, forecast_var_np_pydmd = solver.solver.forecast(t)
        assert isinstance(forecast_np, np.ndarray), (
            "Expected the mean forecast to be a np.ndarray, "
            f"but got {type(forecast_np)} instead."
        )
        assert isinstance(forecast_var_np, np.ndarray), (
            "Expected the forecast variance to be a np.ndarray, "
            f"but got {type(forecast_var_np)} instead."
        )
        np.testing.assert_allclose(
            forecast_np.real,
            forecast_np_pydmd.real,
            err_msg="Expected the mean forecast to match the output of PyDMD.",
        )
        np.testing.assert_allclose(
            forecast_var_np.real,
            forecast_var_np_pydmd.real,
            err_msg="Expected the forecast variance to match the output of PyDMD.",
        )

        # with Dask
        np.random.seed(42)
        forecast_da, forecast_var_da = solver._predict(t, use_dask=True)
        assert isinstance(forecast_da, da.Array), (
            "Expected the mean forecast to be a da.Array, "
            f"but got {type(forecast_da)} instead."
        )
        assert isinstance(forecast_var_da, da.Array), (
            "Expected the forecast variance to be a da.Array, "
            f"but got {type(forecast_var_da)} instead."
        )
        forecast_np, forecast_var_np = forecast_da.compute(), forecast_var_da.compute()
        np.testing.assert_allclose(
            forecast_np.real,
            forecast_np_pydmd.real,
            err_msg="Expected the mean forecast to match the output of PyDMD.",
        )
        np.testing.assert_allclose(
            forecast_var_np.real,
            forecast_var_np_pydmd.real,
            err_msg="Expected the forecast variance to match the output of PyDMD.",
        )


@pytest.mark.parametrize("solver", [optdmd, optdmd_bagging])
def test_forecast(solver):
    """Test for the forecast() method."""
    expected_forecast_shape = (generator.u.shape[0], 10)
    expected_forecast_dims = (generator.u.dims[0], solver.time_dimension)
    if solver.num_trials == 0:
        forecast = solver.forecast(forecast_span="10 s", dt="1 s")
        assert isinstance(forecast, xr.DataArray), (
            "Expected 'forecast' to be of type 'xr.DataArray', "
            f"but got {type(forecast)} instead."
        )
        assert forecast.dims == expected_forecast_dims, (
            f"Expected 'forecast' to have dimensions {expected_forecast_dims}, "
            f"but got {forecast.dims} instead."
        )
        assert forecast.shape == expected_forecast_shape, (
            f"Expected 'forecast' to have shape {expected_forecast_shape}, "
            f"but got {forecast.shape}."
        )
    else:
        forecast, forecast_var = solver.forecast(forecast_span="10 s", dt="1 s")
        assert isinstance(forecast, xr.DataArray), (
            "Expected 'forecast' to be of type 'xr.DataArray', "
            f"but got {type(forecast)} instead."
        )
        assert forecast.dims == expected_forecast_dims, (
            f"Expected 'forecast' to have dimensions {expected_forecast_dims}, "
            f"but got {forecast.dims} instead."
        )
        assert forecast.shape == expected_forecast_shape, (
            f"Expected 'forecast' to have shape {expected_forecast_shape}, "
            f"but got {forecast.shape}."
        )
        assert isinstance(forecast_var, xr.DataArray), (
            "Expected 'forecast_var' to be of type 'xr.DataArray', "
            f"but got {type(forecast)} instead."
        )
        assert forecast_var.dims == expected_forecast_dims, (
            f"Expected 'forecast_var' to have dimensions {expected_forecast_dims}, "
            f"but got {forecast.dims} instead."
        )
        assert forecast_var.shape == expected_forecast_shape, (
            f"Expected 'forecast_var' to have shape {expected_forecast_shape}, "
            f"but got {forecast.shape}."
        )
