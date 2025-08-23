import dask.array as da
import numpy as np
import pytest
import xarray as xr
from test_svd import make_dataarray

from svdrom.pod import POD


@pytest.fixture()
def pod_data():
    """
    Provides an xarray.DataArray with a clear low-rank structure,
    making it suitable for testing reconstruction.
    """
    n_snapshots = 200
    n_points = 1000
    n_rank = 10

    modes_np = np.random.randn(n_points, n_rank).astype("float32")
    coeffs_np = np.sin(np.linspace(0, 4 * np.pi, n_snapshots))[:, None] * np.cos(
        np.linspace(0, np.pi, n_rank)
    )[None, :].astype("float32")
    low_rank_data = coeffs_np @ modes_np.T

    noise = (np.random.rand(n_snapshots, n_points).astype("float32") - 0.5) * 0.1
    data = low_rank_data + noise

    return xr.DataArray(
        data,
        dims=["time", "space"],
        coords={"time": np.arange(n_snapshots), "space": np.arange(n_points)},
    )


def test_pod_initialization():
    """Test that the POD class initializes correctly."""
    pod = POD(n_modes=10, algorithm="randomized")
    assert pod.n_modes == 10
    assert pod.algorithm == "randomized"
    assert pod.modes is None
    assert pod.coeffs is None
    assert pod.energy is None
    assert pod.mean_field is None
    assert pod.s is None


@pytest.mark.parametrize("matrix_type", ["tall-and-skinny", "short-and-fat"])
@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_pod_fit_attributes_and_shapes(matrix_type, algorithm):
    """Test the types, shapes, and metadata of POD attributes after fitting."""
    X = make_dataarray(matrix_type)
    n_samples, n_features = X.shape
    time_dim, space_dim = X.dims
    n_modes = 15

    pod = POD(n_modes, algorithm=algorithm)
    pod.fit(X, dim=time_dim)

    assert isinstance(pod.mean_field, xr.DataArray)
    assert isinstance(pod.modes, xr.DataArray)
    assert isinstance(pod.coeffs, xr.DataArray)
    assert isinstance(pod.energy, np.ndarray)
    assert isinstance(pod.s, np.ndarray)

    assert pod.mean_field.shape == (n_features,)
    assert pod.modes.shape == (n_features, n_modes)
    assert pod.coeffs.shape == (n_samples, n_modes)
    assert pod.energy.shape == (n_modes,)
    assert pod.s.shape == (n_modes,)

    assert pod.coeffs.dims == (time_dim, "modes")
    assert pod.modes.dims == (space_dim, "modes")
    assert "modes" in pod.coeffs.coords and time_dim in pod.coeffs.coords
    assert "modes" in pod.modes.coords and space_dim in pod.modes.coords


@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_pod_calculation_correctness(pod_data, algorithm):
    """Verify the POD results against the reference NumPy implementation."""
    X_xr = pod_data.chunk({"time": -1, "space": "auto"})
    X_np = X_xr.values
    n_snapshots, n_features = X_np.shape
    n_modes = 20

    X_mean_np = np.mean(X_np, axis=0)
    X_fluc_np = X_np - X_mean_np
    X_scaled_np = X_fluc_np / np.sqrt(n_snapshots)
    u_np, s_np, vh_np = np.linalg.svd(X_scaled_np, full_matrices=False)

    modes_np = vh_np.T[:, :n_modes]
    coeffs_np = u_np[:, :n_modes] * s_np[:n_modes]

    svd_kwargs = {}
    if algorithm == "randomized":
        svd_kwargs["n_power_iter"] = 8

    pod = POD(n_modes, algorithm=algorithm)
    pod.fit(X_xr, dim="time")

    rtol = 1e-3

    recon_fluc_pod = pod.coeffs.values @ pod.modes.values.T
    recon_fluc_np = coeffs_np @ modes_np.T

    assert np.allclose(
        recon_fluc_pod, recon_fluc_np, rtol=rtol
    ), "Reconstructed fluctuations from POD do not match the reference NumPy calculation."


def test_pod_reconstruction_and_transform(pod_data):
    """Test data reconstruction and transformation onto the POD basis."""
    X = pod_data.chunk({"time": -1, "space": "auto"})
    n_modes = 15
    pod = POD(n_modes, algorithm="randomized")
    pod.fit(X, dim="time")

    coeffs_transformed = pod.transform(X)
    assert isinstance(
        coeffs_transformed, xr.DataArray
    ), "Transform result should be a DataArray"
    assert (
        coeffs_transformed.shape == pod.coeffs.shape
    ), "Shape of transformed coeffs should match fitted coeffs"

    assert np.allclose(
        np.abs(coeffs_transformed.values), np.abs(pod.coeffs.values), rtol=1e-3
    ), "Transformed coefficients do not match fitted coefficients within tolerance."

    X_reconstructed = pod.reconstruct(n_modes=n_modes)
    assert isinstance(
        X_reconstructed, xr.DataArray
    ), "Reconstruction result should be a DataArray"

    assert (
        X_reconstructed.dims == X.dims
    ), f"Expected dims {X.dims} but got {X_reconstructed.dims}"
    assert (
        X_reconstructed.shape == X.shape
    ), "Shape of reconstructed data should match original"
    assert list(X_reconstructed.coords.keys()) == list(
        X.coords.keys()
    ), "Coordinates should match original"

    error = da.linalg.norm(X.data - X_reconstructed.data) / da.linalg.norm(X.data)
    computed_error = error.compute()
    assert computed_error < 0.1, f"Reconstruction error {computed_error} was too high."
