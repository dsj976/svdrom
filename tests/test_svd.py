import dask.array as da
import numpy as np
import pytest
import xarray as xr

from svdrom.svd import TruncatedSVD


def make_dataarray(matrix_type: str) -> xr.DataArray:
    """Make a Dask-backed DataArray with random data of
    specified matrix type. The matrix type can be one of:
    - "tall-and-skinny": More samples than features.
    - "short-and-fat": More features than samples.
    - "square": Equal number of samples and features.

    Chunks are set to test that the TruncatedSVD can handle
    them correctly.
    """
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
    """Test basic functionality of TruncatedSVD."""
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
    )
    for attr in expected_attrs:
        assert hasattr(tsvd, attr), f"TruncatedSVD should have attribute '{attr}'."

    X = make_dataarray("tall-and-skinny")
    tsvd.fit(X)
    assert isinstance(
        tsvd.u, xr.DataArray
    ), f"u should be an xarray DataArray, got {type(tsvd.u)}."
    assert isinstance(tsvd.u.data, np.ndarray), (
        "u should be a xarray DataArray with numpy ndarray data, "
        f"got {type(tsvd.u.data)}."
    )
    assert isinstance(
        tsvd.v, xr.DataArray
    ), f"v should be an xarray DataArray, got {type(tsvd.v)}."
    assert isinstance(tsvd.v.data, np.ndarray), (
        "v should be a xarray DataArray with numpy ndarray data, "
        f"got {type(tsvd.v.data)}."
    )
    assert isinstance(
        tsvd.s, np.ndarray
    ), f"s should be a numpy ndarray, got {type(tsvd.s)}."
    assert isinstance(tsvd.explained_var_ratio, np.ndarray), (
        "explained_var_ratio should be a numpy ndarray, "
        f"got {type(tsvd.explained_var_ratio)}."
    )
    assert np.all(
        tsvd.explained_var_ratio > 0
    ), "explained_var_ratio should contain values greater than 0."
    assert np.all(
        tsvd.explained_var_ratio < 1
    ), "explained_var_ratio should contain values less than 1."


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat", "square"])
@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_matrix_types(matrix_type, algorithm):
    """Test TruncatedSVD with different matrix shapes."""
    X = make_dataarray(matrix_type)
    n_samples, n_features = X.shape
    n_components = 10
    tsvd = TruncatedSVD(
        n_components=n_components,
        algorithm=algorithm,
    )
    tsvd.fit(X)
    X_dims = list(X.dims)
    X_coords = list(X.coords)
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
    u_dims = tuple(tsvd.u.dims)
    u_coords = tuple(tsvd.u.coords)
    v_dims = tuple(tsvd.v.dims)
    v_coords = tuple(tsvd.v.coords)
    assert u_dims == (
        X_dims[0],
        "components",
    ), f"u should have dimensions ({X_dims[0]}, 'components'), got {u_dims}."
    assert all(
        u_coord in X_coords for u_coord in u_coords if u_coord != "components"
    ), f"u should have all coordinates from X except 'components', got {u_coords}."
    assert "components" in u_coords, "u should have 'components' coordinate."
    assert v_dims == (
        "components",
        X_dims[1],
    ), f"v should have dimensions ('components', {X_dims[1]}), got {v_dims}."
    assert all(
        v_coord in X_coords for v_coord in v_coords if v_coord != "components"
    ), f"v should have all coordinates from X except 'components', got {v_coords}."
    assert "components" in v_coords, "v should have 'components' coordinate."


@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_orthogonality(algorithm):
    """Test orthogonality of u and v matrices."""
    X = make_dataarray("tall-and-skinny")
    n_components = 10
    tsvd = TruncatedSVD(n_components=n_components, algorithm=algorithm)
    tsvd.fit(X)

    identity_k = np.eye(tsvd.n_components, dtype=np.float32)
    u, v = tsvd.u.data, tsvd.v.data
    u_ortho = u.T @ u
    v_ortho = v @ v.T

    assert np.allclose(
        u_ortho, identity_k, atol=1e-5
    ), "u.T @ u is not close to identity."
    assert np.allclose(
        v_ortho, identity_k, atol=1e-5
    ), "v @ v.T is not close to identity."


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat"])
def test_transform(matrix_type):
    """Test the transform method of TruncatedSVD."""
    X = make_dataarray(matrix_type)
    n_components = 10
    tsvd = TruncatedSVD(n_components=n_components)
    tsvd.fit(X)

    X_t = tsvd.transform(X)
    assert isinstance(
        X_t, xr.DataArray
    ), "Transformed data should be an xarray DataArray."
    assert isinstance(X_t.data, np.ndarray), (
        "Transformed data should have numpy ndarray as data, " f"got {type(X_t.data)}."
    )
    assert X_t.shape == (X.shape[0], n_components), (
        f"Transformed data should have shape ({X.shape[0]}, {n_components}), "
        f"but got {X_t.shape}."
    )
    assert np.allclose(
        tsvd.u * tsvd.s, X_t, atol=1e-5
    ), "Transformed data does not match u * s."


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat"])
def test_reconstruct_snapshot(matrix_type):
    """Test the reconstruct_snapshot method of TruncatedSVD."""
    X = make_dataarray(matrix_type)
    n_components = 10
    tsvd = TruncatedSVD(n_components=n_components)
    tsvd.fit(X)

    X_r = tsvd.reconstruct_snapshot(0)
    assert isinstance(
        X_r, xr.DataArray
    ), f"Reconstructed snapshot should be an xarray DataArray, got {type(X_r)}."
    assert isinstance(X_r.data, np.ndarray), (
        "Reconstructed snapshot should have numpy ndarray as data, "
        f"got {type(X_r.data)}."
    )
    assert X_r.shape == (
        tsvd.u.shape[0],
    ), f"Reconstructed snapshot should have shape ({tsvd.u.shape[0]}), got {X_r.shape}."


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat"])
@pytest.mark.parametrize("compute_initially", [True, False])
def test_reconstruct_size_based_computation(matrix_type, compute_initially):
    """Test the reconstruct method's size-based computation (NumPy vs Dask)."""
    X = make_dataarray(matrix_type)
    n_components = 10
    tsvd = TruncatedSVD(
        n_components=n_components,
        compute_u=compute_initially,
        compute_v=compute_initially,
    )
    tsvd.fit(X)

    # --- 1. Test NumPy path (in-memory) ---
    # Force NumPy path by setting a very high memory limit
    X_recon_np = tsvd.reconstruct(memory_limit_bytes=1e12)  # 1 TB limit

    assert isinstance(
        X_recon_np.data, np.ndarray
    ), "Reconstruction should be NumPy-backed for a large memory limit."
    assert X_recon_np.shape == X.shape, "Reconstructed shape must match original."
    assert X_recon_np.dims == X.dims, "Reconstructed dims must match original."
    # Verify coordinates are preserved
    assert X_recon_np.coords["samples"].equals(X.coords["samples"])
    assert X_recon_np.coords["time"].equals(X.coords["time"])

    # --- 2. Test Dask path (out-of-memory) ---
    # Force Dask path by setting a very low memory limit
    X_recon_da = tsvd.reconstruct(memory_limit_bytes=1)  # 1 Byte limit

    assert isinstance(
        X_recon_da.data, da.Array
    ), "Reconstruction should be Dask-backed for a small memory limit."
    assert X_recon_da.shape == X.shape, "Lazy reconstructed shape must match original."
    assert X_recon_da.dims == X.dims, "Lazy reconstructed dims must match original."

    # --- 3. Verify correctness of both results ---
    # Compute the original and Dask-reconstructed data to compare
    X_original_np = X.compute()
    X_recon_da_computed = X_recon_da.compute()

    # The NumPy-path result and the computed Dask-path result should be identical
    np.testing.assert_allclose(
        X_recon_np.data,
        X_recon_da_computed,
        rtol=1e-5,
        err_msg="NumPy and Dask path reconstructions should be nearly identical.",
    )

    # Both reconstructions should be a close approximation of the original data
    # We use a loose tolerance because it's a low-rank approximation
    reconstruction_error = np.linalg.norm(
        X_original_np - X_recon_np.data
    ) / np.linalg.norm(X_original_np)
    assert reconstruction_error < 0.5, "Reconstruction error is unexpectedly high."


def test_reconstruct_before_fit_raises_error():
    """Test that calling reconstruct() before fit() raises a ValueError."""
    tsvd = TruncatedSVD(n_components=5)
    with pytest.raises(
        ValueError, match="The SVD model must be fitted before calling reconstruct"
    ):
        tsvd.reconstruct()
