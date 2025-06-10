import xarray as xr

from svdrom.logger import setup_logger

logger = setup_logger("I/O", "io.log")


class DataHandler:
    def open_dataset(self, filename: str, chunks: str | dict | int = "auto"):
        """
        Opens a compatible file as a Xarray.DataSet (can contain multiple variables).

        Parameters
        ----------
        filename : str
            Path to the file to be opened.
        chunks : str, dict, or int, optional
            Chunking strategy for dask-backed arrays.
            Can be one of:
              - "auto" (default): will use Dask auto chunking.
              - a dictionary mapping dimension names to chunk sizes,
                e.g. `chunks={"time": 10}`.
              - `-1`: loads the data with Dask using a single chunk.

        Notes
        -----
        The opened DataSet is assigned to the `ds` attribute of the instance.
        """
        try:
            logger.info("Opening Xarray.DataSet from %s.", filename)
            self.ds = xr.open_dataset(filename, chunks=chunks)
        except Exception as e:
            msg = f"Error opening {filename} as Xarray.DataSet."
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def open_dataarray(self, filename: str, chunks: str | dict | int = "auto"):
        """
        Opens a compatible file as a Xarray.DataArray (containing a single variable).

        Parameters
        ----------
        filename : str
            Path to the file to be opened.
        chunks : str, dict, or int, optional
            Chunking strategy for dask-backed arrays.
            Can be one of:
              - "auto" (default): will use Dask auto chunking.
              - a dictionary mapping dimension names to chunk sizes,
                e.g. `chunks={"time": 10}`.
              - `-1`: loads the data with Dask using a single chunk.

        Notes
        -----
        The opened DataArray is assigned to the `da` attribute of the instance.
        """
        try:
            logger.info("Opening Xarray.DataArray from %s.", filename)
            self.da = xr.open_dataarray(filename, chunks=chunks)
        except Exception as e:
            msg = f"Error opening {filename} as Xarray.DataArray."
            logger.exception(msg)
            raise RuntimeError(msg) from e
