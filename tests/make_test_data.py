import numpy as np
import xarray as xr


class DataGenerator:
    """
    A utility class for generating synthetic multi-dimensional datasets and data arrays
    containing randomly generated data for testing and development purposes.
    Attributes:
        x (np.ndarray): 1D array representing the x-coordinate values.
        y (np.ndarray): 1D array representing the y-coordinate values.
        z (np.ndarray): 1D array representing the z-coordinate values.
        t (np.ndarray): 1D array representing the time values.
        vars (list): List of variable names to generate data for.
        seed (int | None): Seed for the random number generator for reproducibility.
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
        seed: int | None = 1234,
    ):
        self.x = x if x else np.arange(0, 10)
        self.y = y if y else np.arange(0, 10)
        self.z = z if z else np.arange(0, 5)
        self.t = t if t else np.arange(0, 20)
        self.vars = vars if vars else ["U", "V", "W"]
        self.rng = np.random.default_rng(seed)

    def generate_dataset(self):
        data = {}
        for var in self.vars:
            data[var] = {}
            data[var]["data"] = self.rng.random(
                (self.x.shape[0], self.y.shape[0], self.z.shape[0], self.t.shape[0])
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
        data["data"] = self.rng.random(
            (self.x.shape[0], self.y.shape[0], self.z.shape[0], self.t.shape[0])
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


class SignalGenerator:
    def __init__(
        self,
        nx=100,
        nt=500,
        x_min=0,
        x_max=10,
        t_min=0,
        t_max=50,
    ):
        """A class to generate synthetic coherent spatio-temporal signals
        for testing purposes.

        Parameters
        ----------
        nx : int
            Number of spatial points. Default is 100.
        nt : int
            Number of temporal points. Default is 500.
        x_min : float
            Minimum spatial coordinate. Default is 0.
        x_max : float
            Maximum spatial coordinate. Default is 10.
        t_min : float
            Minimum temporal coordinate. Default is 0.
        t_max : float
            Maximum temporal coordinate. Default is 50.
        """
        self.x = np.linspace(x_min, x_max, nx)
        self.t = np.linspace(t_min, t_max, nt)
        self.X, self.T = np.meshgrid(self.x, self.t)
        data = {}
        data["coords"] = {
            "x": {"dims": ("x"), "data": self.x},
            "time": {"dims": ("time"), "data": self.t},
        }
        data["dims"] = ("time", "x")
        data["data"] = np.zeros(self.X.shape)
        data["name"] = "signal"
        self.da = xr.DataArray.from_dict(data)
        self.components = []

    def add_sinusoid1(
        self, a: float = 1, k: float = 0.1, omega: float = 1, gamma: float = 0
    ) -> "SignalGenerator":
        """Generate a sinusoidal signal of the form:
        a*sin(k*x - omega*t)*exp(gamma*t)

        Parameters
        ----------
        a : float, optional
            Amplitude of the sinusoidal signal, by default 1
        k : float, optional
            Spatial frequency of the signal, by default 0.1
        omega : float, optional
            Temporal frequency of the signal, by default 1
        gamma : float, optional
            Temporal decay rate of the signal, by default 0
        """
        signal = np.sin(k * self.X - omega * self.T) * np.exp(gamma * self.T)
        spatial_norm = np.linalg.norm(signal, axis=-1, ord=2)
        signal = signal / spatial_norm[:, None]
        signal = a * signal
        self.da = self.da.copy(data=self.da.values + signal)
        inputs_dict = {
            "type": "sinusoid1",
            "a": a,
            "k": k,
            "omega": omega,
            "gamma": gamma,
        }
        self.components.append(inputs_dict)
        return self

    def add_sinusoid2(
        self, a: float = 1, k: float = 0.2, omega: float = 1, c: float = 0
    ) -> "SignalGenerator":
        """Generate a sinusoidal signal of the form:
        a*(exp(-k*(x+c)^2)*cos(omega*t)

        Parameters
        ----------
        a : float, optional
            Amplitude (area under the curve) of the signal, by default 1
        k : float, optional
            Spatial exponential decay rate of the signal, by default 0.2
        omega : float, optional
            Temporal frequency of the signal, by default 1
        c : float, optional
            Offset of the signal, by default 0
        """
        spatial_signal = np.exp(-k * (self.X + c) ** 2)
        area = np.trapezoid(spatial_signal, self.x, axis=-1)[
            0
        ]  # Compute the area under the curve
        signal = a * spatial_signal / area * np.cos(omega * self.T)
        self.da = self.da.copy(data=self.da.values + signal)
        inputs_dict = {"type": "sinusoid2", "a": a, "k": k, "omega": omega, "c": c}
        self.components.append(inputs_dict)
        return self

    def add_noise(
        self, noise_std: float = 0.1, random_seed: int | None = None
    ) -> "SignalGenerator":
        """Add Gaussian noise to the signal.

        Parameters
        ----------
        noise_std : float, optional
            Standard deviation of the Gaussian noise, by default 0.1
        random_seed : int | None, optional
            Random seed for reproducibility, by default None
        """
        rng = np.random.default_rng(random_seed)
        noise = rng.normal(0, noise_std, self.X.shape)
        self.da = self.da.copy(data=self.da.values + noise)
        return self

    def _generate_signal(
        self, noise_std: float = 0.2, random_seed: int | None = None
    ) -> "SignalGenerator":
        """Generate signal with three superimposed sinusoids and
        white noise.
        """
        self.add_sinusoid1(a=2, omega=0.5, k=1.5)
        self.add_sinusoid2(a=3, omega=2.5, c=1.5, k=0.5)
        self.add_sinusoid2(a=4, omega=5, c=-1.5, k=0.5)
        self.add_noise(noise_std=noise_std, random_seed=random_seed)
        return self

    def compute_svd_results(
        self,
        n_components: int = -1,
        noise_std: float = 0.2,
        random_seed: int | None = None,
    ) -> "SignalGenerator":
        """Compute the U, S and V matrices resulting from the SVD
        of the signal. If no signal has been computed, a predefined
        signal consisting of three sinusoids and white noise is
        generated. You can control the white noise generation with
        noise_std and random_seed (set to an integer for reproducibility).
        If n_components is -1 (the default), all SVD components are returned.
        """
        if not self.components:
            self._generate_signal(noise_std=noise_std, random_seed=random_seed)
        n_components = (
            min(len(self.x), len(self.t)) if n_components == -1 else n_components
        )
        da = self.da.transpose("x", "time")
        da = da.rename({"x": "samples"})

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

        return self
