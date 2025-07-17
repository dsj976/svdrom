from collections.abc import Sequence

import numpy as np
import xarray as xr


def variable_spatial_stack(
    X: xr.Dataset | xr.DataArray, dims: Sequence[str]
) -> xr.DataArray:
    """Stack multiple dimensions of an input Xarray Dataset or
    DataArray into a single new 'samples' dimension.

    Parameters
    ----------
    X (xr.Dataset | xr.DataArray): the input data to stack.
    dims (Iterable[str]): the dimensions to stack together into a
    single spatial dimension (e.g. ('x', 'y') or ('lat', 'lon')).

    Returns
    -------
    xr.DataArray: The array with the specified dimensions stacked
    into a single spatial dimension.

    Notes
    -----
    If the input Xarray object is a Dataset containing multiple variables,
    all variables will be stacked together along the new 'samples' dimension.
    The resulting DataArray will have a 'samples' dimension that combines
    the specified spatial dimensions (and the variable dimension if the input
    was a Dataset).
    The returned xarray object is Dask-backed and lazy if the input
    is Dask-backed.
    """

    dims = list(dims)
    if isinstance(X, xr.Dataset):
        X = X.to_dataarray(dim="variable")
        dims.insert(0, "variable")

    all_dims: list[str] = list(map(str, X.sizes.keys()))
    if not all(dim in all_dims for dim in dims):
        msg = (
            f"Some dimensions {dims} are not present in the input array "
            f"with dimensions {all_dims}."
        )
        raise ValueError(msg)

    return X.stack(samples=dims)


class StandardScaler:
    """Preprocessing class for scaling Xarray datasets or data arrays
    by removing the mean and optionally scaling by the standard deviation
    along a specified dimension.

    Attributes
    ----------
    mean (xr.DataArray | xr.Dataset): the mean values computed along the
    specified dimension.
    std (xr.DataArray | xr.Dataset): the standard deviation values computed
    along the specified dimension.
    with_std (bool): whether the data has been scaled to unit variance.
    """

    def __init__(self):
        self._mean = xr.DataArray(np.zeros_like(0))
        self._std = xr.DataArray(np.zeros_like(0))
        self._with_std = False

    @property
    def mean(self) -> xr.DataArray | xr.Dataset:
        """Mean (read-only)."""
        return self._mean

    @property
    def std(self) -> xr.DataArray | xr.Dataset:
        """Standard deviation (read-only)."""
        return self._std

    @property
    def with_std(self) -> bool:
        """Whether data scaled to unit variance or not (read_only)."""
        return self._with_std

    def scale(
        self,
        X: xr.Dataset | xr.DataArray,
        dim: str = "time",
        with_std: bool = False,
    ):
        """Scales the input xarray Dataset or DataArray by removing
        the mean and optionally dividing by the standard deviation.

        Parameters:
        -----------
        X (xr.Dataset | xr.DataArray): the input xarray object to be scaled.
        dim (str): the dimension along which to compute the mean and
        standard deviation. Default is "time".
        with_std (bool): if True, scales the data by dividing by the standard
        deviation after subtracting the mean. Default is False.

        Returns:
        --------
        xr.Dataset | xr.DataArray: the scaled xarray object with the mean removed,
        and optionally divided by the standard deviation.

        Note
        ----
        The mean and standard deviation are computed eagerly and stored as NumPy-backed
        xarray objects. The returned xarray object is Dask-backed and lazy if the input
        is Dask-backed.
        """
        self._with_std = with_std
        self._mean = X.mean(dim=dim)
        self._mean = self._mean.compute()
        if self._with_std:
            self._std = X.std(dim=dim)
            self._std = self._std.compute()
            return (X - self._mean) / self._std
        return X - self._mean
