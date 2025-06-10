import os
import shutil

import netCDF4  # noqa: F401
import pytest
import xarray as xr
from make_test_data import DataGenerator

from svdrom.io import DataHandler


class TestDataHandler:
    @classmethod
    def setup_class(cls):
        cls.data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data", "io"
        )
        cls.data_generator = DataGenerator()
        cls.data_handler = DataHandler()

        # create clean directories
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)
        os.makedirs(cls.data_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.data_dir)

    @pytest.fixture(autouse=True)
    def _make_test_data(self):
        print("Generating synthetic Xarray.DataSet...")
        self.data_generator.generate_dataset()
        print("Generating synthetic Xarray.DataArray...")
        self.data_generator.generate_dataarray()
        return

    @pytest.mark.parametrize(
        ("filetype", "ds_ext", "da_ext", "ds_save", "da_save"),
        [
            (
                "zarr",
                "dataset.zarr",
                "dataarray.zarr",
                lambda ds, path: ds.to_zarr(path, zarr_format=2),
                lambda da, path: da.to_zarr(path, zarr_format=2),
            ),
            (
                "netcdf",
                "dataset.nc",
                "dataarray.nc",
                lambda ds, path: ds.to_netcdf(path, format="NETCDF4"),
                lambda da, path: da.to_netcdf(path, format="NETCDF4"),
            ),
        ],
    )
    def test_open_formats(self, filetype, ds_ext, da_ext, ds_save, da_save):
        ds_path = os.path.join(self.data_dir, ds_ext)
        da_path = os.path.join(self.data_dir, da_ext)
        ds_save(self.data_generator.ds, ds_path)
        da_save(self.data_generator.da, da_path)

        self.data_handler.open_dataset(ds_path)
        self.data_handler.open_dataarray(da_path)

        assert isinstance(
            self.data_handler.ds, xr.Dataset
        ), f"Expected xarray.Dataset, got {type(self.data_handler.ds)}."
        assert isinstance(
            self.data_handler.da, xr.DataArray
        ), f"Expected xarray.DataArray, got {type(self.data_handler.da)}."

        assert self.data_generator.ds.equals(
            self.data_handler.ds
        ), f"Xarray Datasets should be equal for {filetype}."
        assert self.data_generator.da.equals(
            self.data_handler.da
        ), f"Xarray DataArrays should be equal for {filetype}."
