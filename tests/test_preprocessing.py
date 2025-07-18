import pytest
from make_test_data import DataGenerator

from svdrom.preprocessing import variable_spatial_stack

generator = DataGenerator()
generator.generate_dataarray()
generator.generate_dataset()


@pytest.mark.parametrize("X", [generator.da, generator.ds])
def test_variable_spatial_stack(X):
    X_stacked = variable_spatial_stack(X, dims=("x", "y", "z"))
    dims = sorted(map(str, X_stacked.sizes.keys()))
    expected_dims = sorted(["time", "samples"])
    assert (
        dims == expected_dims
    ), f"Expected dimensions after stacking are {expected_dims}, got {dims}."
