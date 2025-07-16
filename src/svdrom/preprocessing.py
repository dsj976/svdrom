from collections.abc import Sequence

import dask.array as da
import numpy as np
import xarray as xr
from dask_ml.preprocessing import StandardScaler as ss


class ArrayPreprocessor:
    def spatial_stack(
        self, X: xr.Dataset | xr.DataArray, dims: Sequence[str]
    ) -> xr.Dataset | xr.DataArray:
        """Given an input array, create a single spatial dimension
        by stacking multiple dimensions together. This operation is lazy.

        Parameters
        ----------
        X (xr.Dataset | xr.DataArray): the input array to stack.
        dims (Iterable[str]): the dimensions to stack together into a
        single spatial dimension.

        Returns
        -------
        xr.Dataset | xr.DataArray: The array with the specified
        dimensions stacked into a single spatial dimension.

        Notes
        -----
        This method preserves the data type of the input. If a DataArray is provided,
        a DataArray is returned. If a Dataset is provided, a Dataset is returned.
        """

        all_dims: list[str] = list(map(str, X.sizes.keys()))
        if not all(dim in all_dims for dim in dims):
            msg = (
                f"Some dimensions {dims} are not present in the input array "
                f"with dimensions {all_dims}."
            )
            raise ValueError(msg)

        return X.stack(space=dims)

    def variable_stack(self, X: xr.Dataset, variables: Sequence[str]) -> xr.DataArray:
        """Given a xarray.Dataset containing multiple variables,
        return a xarray.DataArray where the specified variables
        have been stacked along the spatial dimension.

        Not yet implemented.
        """
        pass


class StandardScaler(ArrayPreprocessor):
    def __init__(self):
        self._mean = np.empty_like(0)
        self._std = np.empty_like(0)
        self._with_mean = False
        self._with_std = False
        self._scaler = None

    def fit(
        self,
        X: da.Array | xr.Dataset | xr.DataArray,
        dim="time",
        with_mean: bool = True,
        with_std: bool = False,
        transform: bool = False,
    ):
        pass

    def transform(self):
        pass
