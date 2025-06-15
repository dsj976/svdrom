import dask.array as da
import numpy as np
import pytest

from svdrom.svd import ExactSVD, TruncatedSVD


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
            n_components = 10
            exact_svd = ExactSVD(self.X)
            exact_svd.fit(n_components)
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
            assert exact_svd.u.shape == (
                n_rows,
                n_components,
            ), "The u matrix should have shape (n_rows, n_components)."
            assert exact_svd.v.shape == (
                n_components,
                n_cols,
            ), "The v matrix should have shape (n_components, n_cols)."
            assert exact_svd.s.shape == (
                n_components,
            ), "The s vector should have shape (n_components,)."

    @pytest.mark.parametrize(
        ("matrix_type", "n_rows", "n_cols"),
        [
            ("tall-and-skinny", 10_000, 100),
            ("short-and-fat", 100, 10_000),
            ("square", 1_000, 1_000),
        ],
    )
    def test_truncated_svd(self, matrix_type, n_rows, n_cols):
        self._make_matrix(n_rows, n_cols)
        n_components = 10

        if matrix_type == "square":
            with pytest.raises(
                RuntimeError, match="truncated SVD algorithm can only handle"
            ):
                TruncatedSVD(self.X)
        else:
            truncated_svd = TruncatedSVD(self.X)
            truncated_svd.fit(n_components=n_components)

            assert hasattr(truncated_svd, "u")
            assert hasattr(truncated_svd, "s")
            assert hasattr(truncated_svd, "v")

            u, s, v = truncated_svd.u, truncated_svd.s, truncated_svd.v

            assert isinstance(u, da.Array)
            assert isinstance(s, np.ndarray)
            assert isinstance(v, da.Array)

            assert u.shape == (n_rows, n_components)
            assert s.shape == (n_components,)
            assert v.shape == (n_components, n_cols)

            # check ortho
            identity_k = np.eye(n_components, dtype=np.float32)
            u_ortho = (u.T @ u).compute()
            v_ortho = (v @ v.T).compute()

            assert np.allclose(u_ortho, identity_k, atol=1e-5)
            assert np.allclose(v_ortho, identity_k, atol=1e-5)
