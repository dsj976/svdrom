import dask.array as da
import numpy as np
import pytest
import xarray as xr

from svdrom.svd import TruncatedSVD


def make_dataarray(matrix_type: str) -> xr.DataArray:
    if matrix_type == "tall-and-skinny":
        n_samples = 10_000
        n_features = 100
        X = da.random.random(
            (n_samples, n_features), chunks=(-1, int(n_features / 2))
        ).astype("float32")
    elif matrix_type == "short-and-fat":
        n_samples = 100
        n_features = 10_000
        X = da.random.random(
            (n_samples, n_features), chunks=(int(n_samples / 2), -1)
        ).astype("float32")
    elif matrix_type == "square":
        n_samples = 1_000
        n_features = 1_000
        X = da.random.random(
            (n_samples, n_features), chunks=(int(n_samples / 2), int(n_features / 2))
        ).astype("float32")
    else:
        msg = (
            "Matrix type not supported. "
            "Must be one of: tall-and-skinny, short-and-fat, square."
        )
        raise ValueError(msg)
    coords = {"samples": np.arange(n_samples), "time": np.arange(n_features)}
    dims = list(coords.keys())
    return xr.DataArray(X, dims=dims, coords=coords)


@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_basic(algorithm):
    n_components = 10
    tsvd = TruncatedSVD(
        n_components=n_components,
        algorithm=algorithm,
        compute_var_ratio=True,
    )
    expected_attrs = (
        "u",
        "s",
        "v",
        "explained_var_ratio",
        "aspect_ratio",
    )
    for attr in expected_attrs:
        assert hasattr(tsvd, attr), f"TruncatedSVD should have attribute '{attr}'."

    X = make_dataarray("tall-and-skinny")
    n_samples, n_features = X.shape
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
    assert isinstance(tsvd.explained_var_ratio, np.ndarray), (
        "explained_var_ratio should be a numpy ndarray, "
        f"got {type(tsvd.explained_var_ratio)}."
    )
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
    assert tsvd.explained_var_ratio.shape == (n_components,), (
        f"Shape of explained_var_ratio should be ({n_components},), "
        f"got {tsvd.explained_var_ratio.shape}."
    )


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat", "square"])
@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_matrix_types(matrix_type, algorithm):
    X = make_dataarray(matrix_type)
    n_components = 10
    tsvd = TruncatedSVD(
        n_components=n_components,
        algorithm=algorithm,
    )
    tsvd.fit(X)
