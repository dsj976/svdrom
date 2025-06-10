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

    def test_open_zarr(self):
        self.data_generator.ds.to_zarr(
            os.path.join(self.data_dir, "dataset.zarr"), zarr_format=2
        )
        self.data_generator.da.to_zarr(
            os.path.join(self.data_dir, "dataarray.zarr"), zarr_format=2
        )

        self.data_handler.open_dataset(os.path.join(self.data_dir, "dataset.zarr"))
        self.data_handler.open_dataarray(os.path.join(self.data_dir, "dataarray.zarr"))

        assert isinstance(
            self.data_handler.ds, xr.Dataset
        ), f"Expected xarray.Dataset, got {type(self.data_handler.ds)}"
        assert isinstance(
            self.data_handler.da, xr.DataArray
        ), f"Expected xarray.DataArray, got {type(self.data_handler.da)}"

        assert self.data_generator.ds.equals(
            self.data_handler.ds
        ), "Xarray Datasets should be equal"
        assert self.data_generator.da.equals(
            self.data_handler.da
        ), "Xarray DataArrays should be equal"

    def test_open_netcdf(self):
        self.data_generator.ds.to_netcdf(
            os.path.join(self.data_dir, "dataset.nc"), format="NETCDF4"
        )
        self.data_generator.da.to_netcdf(
            os.path.join(self.data_dir, "dataarray.nc"), format="NETCDF4"
        )

        self.data_handler.open_dataset(os.path.join(self.data_dir, "dataset.nc"))
        self.data_handler.open_dataarray(os.path.join(self.data_dir, "dataarray.nc"))

        assert isinstance(
            self.data_handler.ds, xr.Dataset
        ), f"Expected xarray.Dataset, got {type(self.data_handler.ds)}"
        assert isinstance(
            self.data_handler.da, xr.DataArray
        ), f"Expected xarray.DataArray, got {type(self.data_handler.da)}"

        assert self.data_generator.ds.equals(
            self.data_handler.ds
        ), "Xarray Datasets should be equal"
        assert self.data_generator.da.equals(
            self.data_handler.da
        ), "Xarray DataArrays should be equal"
