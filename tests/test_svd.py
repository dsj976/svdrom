import dask.array as da
import numpy as np
import pytest

from svdrom.svd import ExactSVD


class TestSVD:
    def _make_matrix(self, n_rows, n_cols):
        self.X = da.random.random((n_rows, n_cols)).astype("float32")

    @pytest.mark.parametrize(
        ("matrix_type", "n_rows", "n_cols"),
        [
            ("tall-and-skinny", 10_000, 100),
            ("short-and_fat", 100, 10_000),
            ("square", 1_000, 1_000),
        ],
    )
    def test_exact_svd(self, matrix_type, n_rows, n_cols):
        self._make_matrix(n_rows, n_cols)
        if matrix_type == "square":
            with pytest.raises(RuntimeError):
                exact_svd = ExactSVD(self.X)
        else:
            exact_svd = ExactSVD(self.X)
            exact_svd.fit(n_components=10)
            assert isinstance(exact_svd.u, da.Array), (
                "The u matrix should be of type dask.array.Array, "
                f"not {type(exact_svd.u)}."
            )
            assert isinstance(exact_svd.v, da.Array), (
                "The v matrix should be of type dask.array.Array, "
                f"not {type(exact_svd.v)}."
            )
            assert isinstance(exact_svd.s, np.ndarray), (
                "The s vector should be of type numpy.ndarray, "
                f"not {type(exact_svd.s)}."
            )
