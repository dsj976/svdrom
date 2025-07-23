import dask.array as da
import numpy as np
import pytest
import xarray as xr

from svdrom.svd import TruncatedSVD


def make_dataarray(n_rows, n_cols):
    X = da.random.random((n_rows, n_cols), chunks="auto").astype("float32")
    coords = {"samples": np.arange(n_rows), "time": np.arange(n_cols)}
    dims = list(coords.keys())
    return xr.DataArray(X, dims=dims, coords=coords)


@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_basic(algorithm):
    n_samples = 10_000
    n_features = 100
    n_components = 10
    tsvd = TruncatedSVD(
        n_components=n_components,
        algorithm=algorithm,
        compute_var_ratio=True,
        rechunk=False,
    )
    expected_attrs = (
        "u",
        "s",
        "v",
        "explained_var_ratio",
        "matrix_type",
        "aspect_ratio",
    )
    for attr in expected_attrs:
        assert hasattr(tsvd, attr), f"TruncatedSVD should have attribute '{attr}'."

    X = make_dataarray(n_samples, n_features)
    tsvd.fit(X)
    assert isinstance(
        tsvd.u, xr.DataArray
    ), f"u should be an xarray DataArray, got {type(tsvd.u)}."
    assert isinstance(
        tsvd.v, xr.DataArray
    ), f"v should be an xarray DataArray, got {type(tsvd.v)}."
    assert isinstance(
        tsvd.s, np.ndarray
    ), f"s should be a numpy ndarray, got {type(tsvd.s)}."
    assert tsvd.u.shape == (
        X.shape[0],
        n_components,
    ), f"Shape of u should be ({n_samples}, {n_components}), got {tsvd.u.shape}."
    assert tsvd.v.shape == (
        n_components,
        X.shape[1],
    ), f"Shape of v should be ({n_components}, {n_features}), got {tsvd.v.shape}."
    assert tsvd.s.shape == (
        n_components,
    ), f"Shape of s should be ({n_components},), got {tsvd.s.shape}."
