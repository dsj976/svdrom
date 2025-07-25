import pytest
import xarray as xr
from make_test_data import DataGenerator

from svdrom.preprocessing import StandardScaler, variable_spatial_stack

generator = DataGenerator()
generator.generate_dataarray()
generator.generate_dataset()


@pytest.mark.parametrize("X", [generator.da, generator.ds])
def test_variable_spatial_stack(X: xr.DataArray | xr.Dataset):
    """Test for the variable_spatial_stack function."""
    # check that the output is a xarray DataArray
    X_stacked = variable_spatial_stack(X, dims=("x", "y", "z"))
    assert isinstance(X_stacked, xr.DataArray), (
        "The result of stacking should be a Xarray DataArray, ",
        f"got {type(X_stacked).__name__}.",
    )

    # check that the dimensions are "time" and "samples"
    dims = sorted(map(str, X_stacked.sizes.keys()))
    expected_dims = sorted(["time", "samples"])
    assert (
        dims == expected_dims
    ), f"Expected dimensions after stacking are {expected_dims}, got {dims}."

    # check if shape matches expected shape
    X_stacked = X_stacked.transpose("samples", "time")
    len_x, len_y, len_z = len(X.x), len(X.y), len(X.z)
    if isinstance(X, xr.DataArray):
        expected_shape = (len_x * len_y * len_z, len(X.time))
        assert expected_shape == X_stacked.shape, (
            f"Expected shape for a DataArray after stacking is {expected_shape}, ",
            f"got {X_stacked.shape}.",
        )
    if isinstance(X, xr.Dataset):
        vars = list(X.keys())
        expected_shape = (len_x * len_y * len_z * len(vars), len(X.time))
        assert expected_shape == X_stacked.shape, (
            f"Expected shape for a Dataset after stacking is {expected_shape}, ",
            f"got {X_stacked.shape}.",
        )

    # check that by unstacking we can recover the original
    X_unstacked = X_stacked.unstack()
    if isinstance(X, xr.DataArray):
        xr.testing.assert_allclose(X, X_unstacked, check_dim_order=False)
    if isinstance(X, xr.Dataset):
        X_unstacked = X_unstacked.to_dataset(dim="variable")
        xr.testing.assert_allclose(X, X_unstacked, check_dim_order=False)


@pytest.mark.parametrize("X", [generator.da, generator.ds])
def test_standard_scaler(X: xr.DataArray | xr.Dataset):
    """Test for the StandardScaler class."""
    scaler = StandardScaler()
    X_scaled = scaler(X, with_std=True)

    assert hasattr(scaler, "mean")
    assert hasattr(scaler, "std")
    assert hasattr(scaler, "with_std")

    if isinstance(X, xr.DataArray):
        assert isinstance(
            X_scaled, xr.DataArray
        ), f"Expected a DataArray after scaling, got {type(X_scaled).__name__}."
        assert isinstance(
            scaler.mean, xr.DataArray
        ), f"Expected mean to be a DataArray, got {type(scaler.mean).__name__}."
        assert isinstance(
            scaler.std, xr.DataArray
        ), f"Expected std to be a DataArray, got {type(scaler.std).__name__}."
        if scaler.with_std:
            xr.testing.assert_allclose(X, X_scaled * scaler.std + scaler.mean)
        else:
            xr.testing.assert_allclose(X, X_scaled + scaler.mean)
    if isinstance(X, xr.Dataset):
        assert isinstance(
            X_scaled, xr.Dataset
        ), f"Expected a Dataset after scaling, got {type(X_scaled).__name__}."
        assert isinstance(
            scaler.mean, xr.Dataset
        ), f"Expected mean to be a Dataset, got {type(scaler.mean).__name__}."
        assert isinstance(
            scaler.std, xr.Dataset
        ), f"Expected std to be a Dataset, got {type(scaler.std).__name__}."
        if scaler.with_std:
            xr.testing.assert_allclose(X, X_scaled * scaler.std + scaler.mean)
        else:
            xr.testing.assert_allclose(X, X_scaled + scaler.mean)
