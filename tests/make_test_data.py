import numpy as np
import xarray as xr


class DataGenerator:
    """
    A utility class for generating synthetic multi-dimensional datasets and data arrays
    for testing and development purposes.
    Attributes:
        x (np.ndarray): 1D array representing the x-coordinate values.
        y (np.ndarray): 1D array representing the y-coordinate values.
        z (np.ndarray): 1D array representing the z-coordinate values.
        t (np.ndarray): 1D array representing the time values.
        vars (list): List of variable names to generate data for.
        ds (xr.Dataset): Generated xarray Dataset (after calling generate_dataset()).
        da (xr.DataArray): Generated xarray DataArray(after calling
            generate_dataarray()).
        u (xr.DataArray): Left singular vectors from SVD (after calling
            generate_svd_results()).
        s (np.ndarray): Singular values from SVD (after calling
            generate_svd_results()).
        v (xr.DataArray): Right singular vectors from SVD (after calling
            generate_svd_results()).

    Methods:
        generate_dataset():
            Generates a synthetic xarray.Dataset with random data for each variable
            in `vars`. The dataset includes coordinates for x, y, z, and time.
        generate_dataarray(var: str | None = None):
            Generates a synthetic xarray.DataArray for a specified variable (or the
            first in `vars` by default) with random data and appropriate coordinates.
        generate_svd_results(n_components: int = -1):
            Generates synthetic SVD results (u, s, v) from the DataArray. If
            n_components is -1 (default), all SVD components are returned.
    """

    def __init__(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
        t: np.ndarray | None = None,
        vars: list | None = None,
    ):
        self.x = x if x else np.arange(0, 10)
        self.y = y if y else np.arange(0, 10)
        self.z = z if z else np.arange(0, 5)
        self.t = t if t else np.arange(0, 20)
        self.vars = vars if vars else ["U", "V", "W"]

    def generate_dataset(self):
        data = {}
        for var in self.vars:
            data[var] = {}
            data[var]["data"] = np.random.rand(
                self.x.shape[0], self.y.shape[0], self.z.shape[0], self.t.shape[0]
            )
            data[var]["dims"] = ("x", "y", "z", "time")
        data["x"] = {"data": self.x, "dims": ("x")}
        data["y"] = {"data": self.y, "dims": ("y")}
        data["z"] = {"data": self.z, "dims": ("z")}
        data["time"] = {"data": self.t, "dims": ("time")}
        self.ds = xr.Dataset.from_dict(data)

    def generate_dataarray(self, var: str | None = None):
        var = var if var else self.vars[0]
        if var not in self.vars:
            msg = f"{var} not in variable list: {self.vars}."
            raise ValueError(msg)
        data = {}
        data["coords"] = {
            "x": {"dims": ("x"), "data": self.x},
            "y": {"dims": ("y"), "data": self.y},
            "z": {"dims": ("z"), "data": self.z},
            "time": {"dims": ("time"), "data": self.t},
        }
        data["dims"] = ("x", "y", "z", "time")
        data["data"] = np.random.rand(
            self.x.shape[0], self.y.shape[0], self.z.shape[0], self.t.shape[0]
        )
        data["name"] = var
        self.da = xr.DataArray.from_dict(data)

    def generate_svd_results(self, n_components: int = -1):
        n_components = len(self.t) if n_components == -1 else n_components
        self.generate_dataarray()
        da = self.da.stack(samples=("x", "y", "z"))
        da = da.transpose("samples", "time")

        u, s, v = np.linalg.svd(da.data, full_matrices=False)
        u, s, v = u[:, :n_components], s[:n_components], v[:n_components, :]

        dims = ["samples", "components"]
        coords = {}
        coords["samples"] = da.coords["samples"]
        coords["components"] = np.arange(n_components)
        self.u = xr.DataArray(u, dims=dims, coords=coords)

        self.s = s

        dims = ["components", "time"]
        coords = {}
        coords["components"] = np.arange(n_components)
        coords["time"] = da.coords["time"]
        self.v = xr.DataArray(v, dims=dims, coords=coords)
