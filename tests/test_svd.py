import dask.array as da
import numpy as np
import pytest

from svdrom.svd import ExactSVD, RandomizedSVD, TruncatedSVD


@pytest.mark.parametrize(
    ("matrix_type", "n_rows", "n_cols"),
    [
        ("tall-and-skinny", 100_000, 100),
        ("short-and-fat", 100, 100_000),
        ("square", 10_000, 5_000),
    ],
)
class TestSVD:
    """
    Test suite for validating the functionality of SVD
    (Singular Value Decomposition) implementations.

    This class provides methods to test SVD algorithms using
    Dask arrays. It checks for correct output types, shapes,
    and expected exceptions for different matrix types
    (tall-and-skinny, short-and-fat or square/nearly square).

    Methods
    -------
    _make_matrix(n_rows, n_cols)
        Helper method to create a random Dask array of the
        specified shape with auto chunking.

    test_exact_svd(matrix_type, n_rows, n_cols)
        Tests the ExactSVD implementation.
        Verifies correct exception handling, output types,
        and shapes.

    test_randomized_svd(matrix_type, n_rows, n_cols)
        Tests the RandomizedSVD implementation.
        Checks output types and shapes for correctness.
    """

    def _make_matrix(self, n_rows, n_cols):
        self.X = da.random.random((n_rows, n_cols), chunks="auto").astype("float32")

    def test_exact_svd(self, matrix_type, n_rows, n_cols):
        print(f"Running exact SVD test for {matrix_type} matrix.")
        self._make_matrix(n_rows, n_cols)
        if matrix_type == "square":
            with pytest.raises(RuntimeError):
                exact_svd = ExactSVD(self.X)
        else:
            n_components = 10
            exact_svd = ExactSVD(self.X)
            exact_svd.fit(n_components, transform=True)

            assert hasattr(
                exact_svd, "u"
            ), "The exact_svd object should have attribute 'u'."
            assert hasattr(
                exact_svd, "s"
            ), "The exact_svd object should have attribute 's'."
            assert hasattr(
                exact_svd, "v"
            ), "The exact_svd object should have attribute 'v'."
            assert hasattr(
                exact_svd, "n_components"
            ), "The exact_svd object should have attribute 'n_components'."

            u, s, v = exact_svd.u, exact_svd.s, exact_svd.v
            assert isinstance(
                u, da.Array
            ), f"The u matrix should be of type dask.array.Array, not {type(u)}."
            assert isinstance(
                v, da.Array
            ), f"The v matrix should be of type dask.array.Array, not {type(v)}."
            assert isinstance(
                s, np.ndarray
            ), f"The s vector should be of type numpy.ndarray, not {type(s)}."
            assert u.shape == (
                n_rows,
                n_components,
            ), "The u matrix should have shape (n_samples, n_components)."
            assert v.shape == (
                n_components,
                n_cols,
            ), "The v matrix should have shape (n_components, n_features)."
            assert s.shape == (
                n_components,
            ), "The s vector should have shape (n_components,)."

    def test_randomized_svd(self, matrix_type, n_rows, n_cols):
        print(f"Running randomized SVD test for {matrix_type} matrix.")
        self._make_matrix(n_rows, n_cols)
        n_components = 10
        randomized_svd = RandomizedSVD(self.X)
        randomized_svd.fit(n_components, transform=True)

        assert hasattr(
            randomized_svd, "u"
        ), "The randomized_svd object should have attribute 'u'."
        assert hasattr(
            randomized_svd, "s"
        ), "The randomized_svd object should have attribute 's'."
        assert hasattr(
            randomized_svd, "v"
        ), "The randomized_svd object should have attribute 'v'."
        assert hasattr(
            randomized_svd, "n_components"
        ), "The randomized_svd object should have attribute 'n_components'."

        u, s, v = randomized_svd.u, randomized_svd.s, randomized_svd.v
        assert isinstance(
            u, da.Array
        ), f"The u matrix should be of type dask.array.Array, not {type(u)}."
        assert isinstance(
            v, da.Array
        ), f"The v matrix should be of type dask.array.Array, not {type(v)}."
        assert isinstance(
            s, np.ndarray
        ), f"The s vector should be of type numpy.ndarray, not {type(s)}."
        assert u.shape == (
            n_rows,
            n_components,
        ), (
            f"The u matrix should have shape ({n_rows}, {n_components}), "
            f"but got {u.shape}."
        )
        assert v.shape == (
            n_components,
            n_cols,
        ), (
            f"The v matrix should have shape ({n_components}, {n_cols}), "
            f"but got {v.shape}."
        )
        assert s.shape == (
            n_components,
        ), f"The s vector should have shape ({n_components},), but got {s.shape}."

    def test_truncated_svd(self, matrix_type, n_rows, n_cols):
        print(f"Running truncated SVD test for {matrix_type} matrix.")
        self._make_matrix(n_rows, n_cols)
        n_components = 10

        if matrix_type == "square":
            with pytest.raises(RuntimeError):
                TruncatedSVD(self.X)
        else:
            truncated_svd = TruncatedSVD(self.X)
            truncated_svd.fit(n_components=n_components, transform=True)

            assert hasattr(
                truncated_svd, "u"
            ), "The truncated_svd object should have attribute 'u'."
            assert hasattr(
                truncated_svd, "s"
            ), "The truncated_svd object should have attribute 's'."
            assert hasattr(
                truncated_svd, "v"
            ), "The truncated_svd object should have attribute 'v'."
            assert hasattr(
                truncated_svd, "n_components"
            ), "The truncated_svd object should have attribute 'n_components'."

            u, s, v = truncated_svd.u, truncated_svd.s, truncated_svd.v

            assert isinstance(
                u, da.Array
            ), f"The u matrix should be of type dask.array.Array, not {type(u)}."
            assert isinstance(
                s, np.ndarray
            ), f"The s vector should be of type numpy.ndarray, not {type(s)}."
            assert isinstance(
                v, da.Array
            ), f"The v matrix should be of type dask.array.Array, not {type(v)}."

            assert u.shape == (n_rows, n_components), (
                f"The u matrix should have shape ({n_rows}, {n_components}), "
                f"but got {u.shape}."
            )
            assert s.shape == (n_components,), (
                f"The s vector should have shape ({n_components},), "
                f"but got {s.shape}."
            )
            assert v.shape == (n_components, n_cols), (
                f"The v matrix should have shape ({n_components}, {n_cols}), "
                f"but got {v.shape}."
            )

            # check orthogonality
            identity_k = np.eye(n_components, dtype=np.float32)
            u_ortho = (u.T @ u).compute()
            v_ortho = (v @ v.T).compute()

            assert np.allclose(
                u_ortho, identity_k, atol=1e-5
            ), "u.T @ u is not close to identity."
            assert np.allclose(
                v_ortho, identity_k, atol=1e-5
            ), "v @ v.T is not close to identity."
