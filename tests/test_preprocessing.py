import pytest
import xarray as xr
from make_test_data import DataGenerator

from svdrom.preprocessing import variable_spatial_stack

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
    X = X.transpose("x", "y", "z", "time")
    if isinstance(X, xr.DataArray):
        X_unstacked = X_unstacked.transpose("x", "y", "z", "time")
        assert X.equals(
            X_unstacked
        ), "Unstacking should recover the original DataArray."
    if isinstance(X, xr.Dataset):
        X_unstacked = X_unstacked.transpose("x", "y", "z", "time", "variable")
        X_unstacked = X_unstacked.to_dataset(dim="variable")
        assert X.equals(X_unstacked), "Unstacking should recover the original Dataset."
