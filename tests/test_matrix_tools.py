import unittest

import scipy.sparse
import numpy as np
import scipy
import sympy

from lsrl.matrix_tools import Matrix, MatrixMode, matrix_mode, set_global_matrix_mode
from lsrl.utils import is_fully_symbolic


class TestMatrixGlobalMode(unittest.TestCase):

    def setUp(self) -> None:
        self.values = [[1.0, 0], [0, 1]]

    def test_global_mode_effect(self):
        """Test the effect of setting a global matrix mode."""
        set_global_matrix_mode(MatrixMode.NUMERIC)
        matrix = Matrix(self.values)  # Should default to numeric
        self.assertTrue(scipy.sparse.issparse(matrix.matrix))
        self.assertEqual(matrix.mode, MatrixMode.NUMERIC)

        set_global_matrix_mode(MatrixMode.SYMBOLIC)
        matrix = Matrix(self.values)  # Should default to symbolic
        self.assertIsInstance(matrix.matrix, sympy.SparseMatrix)
        self.assertEqual(matrix.mode, MatrixMode.SYMBOLIC)
        self.assertTrue(is_fully_symbolic(matrix.matrix))

    def test_context_manager_mode(self):
        """Test matrix creation within a matrix_mode context manager."""
        set_global_matrix_mode(MatrixMode.NUMERIC)
        with matrix_mode(MatrixMode.SYMBOLIC):
            matrix = Matrix(self.values)
            self.assertIsInstance(matrix.matrix, sympy.SparseMatrix)
            self.assertEqual(matrix.mode, MatrixMode.SYMBOLIC)
        # Context manager should restore the previous mode
        matrix = Matrix(self.values)

        self.assertTrue(scipy.sparse.issparse(matrix.matrix))
        self.assertEqual(matrix.mode, MatrixMode.NUMERIC)

    def test_global_mode_operations(self):
        """Test operations respect the current global matrix mode."""
        set_global_matrix_mode(MatrixMode.NUMERIC)
        numeric_matrix = Matrix(self.values)
        with matrix_mode(MatrixMode.SYMBOLIC):
            symbolic_matrix = Matrix(self.values)
            with self.assertRaises(ValueError):
                _ = numeric_matrix + symbolic_matrix  # Should fail due to mode mismatch

        # Outside of context manager, both should be numeric again
        set_global_matrix_mode(MatrixMode.NUMERIC)
        numeric_matrix2 = Matrix(self.values)
        result = numeric_matrix + numeric_matrix2
        self.assertTrue(np.allclose(result.matrix.toarray(), np.array(self.values) * 2))


class TestMatrixOperations(unittest.TestCase):

    def setUp(self):
        # Numeric matrices
        self.values = np.array([[1, 3], [2, 1]])

    def test_creation_numeric(self):
        """Test creating a numeric sparse matrix."""
        matrix = Matrix(self.values, mode=MatrixMode.NUMERIC)
        self.assertTrue(scipy.sparse.issparse(matrix.matrix))
        self.assertEqual(matrix.mode, MatrixMode.NUMERIC)

    def test_creation_symbolic(self):
        """Test creating a symbolic sparse matrix."""
        matrix = Matrix(self.values, mode=MatrixMode.SYMBOLIC)
        self.assertIsInstance(matrix.matrix, sympy.SparseMatrix)
        self.assertEqual(matrix.mode, MatrixMode.SYMBOLIC)

    def test_addition(self):
        """Test matrix addition for both modes."""
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(str(mode.name)):
                with matrix_mode(mode):
                    matrix_A = Matrix(self.values)
                    matrix_B = Matrix(-2.1 * self.values)

                    result = matrix_A + matrix_B
                    result_r = matrix_B + matrix_A

                    np.testing.assert_allclose(result.numpy(), self.values - 2.1 * self.values)
                    np.testing.assert_allclose(result_r.numpy(), self.values - 2.1 * self.values)

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))
                        self.assertTrue(is_fully_symbolic(result_r.matrix))

    def test_addition_with_scalars_and_arrays(self):
        """Test matrix addition with different scalar and array types."""
        # Setup initial matrices in numeric and symbolic modes.

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(str(mode.name)):
                with matrix_mode(mode):
                    matrix = Matrix(self.values)

                    # Test data for numeric and symbolic operations.
                    test_values = [
                        1,  # integer
                        -3.5,  # float
                        sympy.pi,  # sympy.Basic
                        Matrix([[1, 1], [-1, -1]]),
                    ]

                    for value in test_values:
                        # Testing __add__
                        if isinstance(value, (int, float)):
                            expected = self.values + value
                        elif isinstance(value, sympy.Basic):
                            expected = self.values + float(value)
                        elif isinstance(value, Matrix):
                            expected = self.values + value.numpy()

                        result = matrix + value
                        result_r = value + matrix

                        np.testing.assert_allclose(result.numpy(), expected)
                        np.testing.assert_allclose(result_r.numpy(), expected)

                        if mode == MatrixMode.SYMBOLIC:
                            self.assertTrue(is_fully_symbolic(result.matrix))
                            self.assertTrue(is_fully_symbolic(result_r.matrix))

    def test_multiplication(self):
        """Test matrix multiplicaiton with different scalar types."""
        # Setup initial matrices in numeric and symbolic modes.

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(str(mode.name)):
                with matrix_mode(mode):
                    matrix = Matrix(self.values)

                    # Test data for numeric and symbolic operations.
                    test_values = [
                        1,  # integer
                        -3.5,  # float
                        sympy.pi,  # sympy.Basic
                    ]

                    for value in test_values:
                        # Testing __mul__
                        if isinstance(value, (int, float)):
                            expected = self.values * value
                        elif isinstance(value, sympy.Basic):
                            expected = self.values * float(value)
                        else:
                            raise ValueError("Unexpected value type")

                        result = matrix * value
                        result_r = value * matrix

                        np.testing.assert_allclose(result.numpy(), expected)
                        np.testing.assert_allclose(result_r.numpy(), expected)

                        if mode == MatrixMode.SYMBOLIC:
                            self.assertTrue(is_fully_symbolic(result.matrix))
                            self.assertTrue(is_fully_symbolic(result_r.matrix))

    def test_subtraction(self):
        """Test matrix subtraction for both modes."""
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(str(mode.name)):
                with matrix_mode(mode):
                    matrix_A = Matrix(self.values)
                    matrix_B = Matrix(1.5 * self.values)

                    result = matrix_A - matrix_B
                    result_r = matrix_B - matrix_A

                    np.testing.assert_allclose(result.numpy(), self.values - 1.5 * self.values)
                    np.testing.assert_allclose(result_r.numpy(), 1.5 * self.values - self.values)

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))
                        self.assertTrue(is_fully_symbolic(result_r.matrix))

    def test_matrix_multiplication(self):
        """Test matrix multiplication for both modes."""
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(str(mode.name)):
                with matrix_mode(mode):
                    matrix_A = Matrix(self.values)
                    matrix_B = Matrix(np.array([[2, 0], [0, 2]]))

                    result = matrix_A @ matrix_B

                    np.testing.assert_allclose(result.numpy(), self.values.dot([[2, 0], [0, 2]]))

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))

    def test_getitem(self):
        """Test element access via getitem for both modes."""
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(str(mode.name)):
                with matrix_mode(mode):

                    # # expect runtime error
                    # with self.assertRaises(IndexError):
                    #     _ = matrix[0, 0]
                    # with self.assertRaises(IndexError):
                    #     __annotations__ = matrix[1, 1]
                    # with self.assertRaises(IndexError):
                    #     _ = matrix[0, :]
                    # with self.assertRaises(IndexError):
                    #     _ = matrix[1]

                    vector = Matrix(self.values[0][:, None])
                    sl1 = vector[0]
                    sl2 = vector[1]
                    sl3 = vector[0:2]
                    sl4 = vector[0:]
                    sl5 = vector[:2]

                    for sl in [sl1, sl2, sl3, sl4, sl5]:
                        self.assertTrue(isinstance(sl, Matrix))

                    np.testing.assert_allclose(sl1.numpy(), [[self.values[0, 0]]])
                    np.testing.assert_allclose(sl2.numpy(), [[self.values[0, 1]]])
                    np.testing.assert_allclose(sl3.numpy(), np.array([self.values[0, :]]).T)
                    np.testing.assert_allclose(sl4.numpy(), np.array([self.values[0, :]]).T)
                    np.testing.assert_allclose(sl5.numpy(), np.array([self.values[0, :]]).T)

                    matrix = Matrix(self.values)
                    sl1 = matrix[0, :]
                    sl2 = matrix[:, 1]
                    sl3 = matrix[1, 1]
                    sl4 = matrix[1:, :]
                    sl5 = matrix[0, 0:]

                    for sl in [sl1, sl2, sl3, sl4, sl5]:
                        self.assertTrue(isinstance(sl, Matrix))

                    np.testing.assert_allclose(sl1.numpy(), self.values[0, :][None, :])
                    np.testing.assert_allclose(sl2.numpy(), self.values[:, 1][:, None])
                    np.testing.assert_allclose(sl3.numpy(), [[self.values[1, 1]]])
                    np.testing.assert_allclose(sl4.numpy(), self.values[1:, :])
                    np.testing.assert_allclose(sl5.numpy(), self.values[0, 0:][None, :])

    def test_hvstack(self):
        """Test horizontal and vertical stacking for both modes."""
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(str(mode.name)):
                with matrix_mode(mode):
                    matrix_A = Matrix(self.values)
                    matrix_B = Matrix(self.values)

                    result_h = Matrix.hstack(matrix_A, matrix_B)
                    result_v = Matrix.vstack(matrix_A, matrix_B)

                    np.testing.assert_allclose(result_h.numpy(), np.hstack([self.values, self.values]))
                    np.testing.assert_allclose(result_v.numpy(), np.vstack([self.values, self.values]))

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result_h.matrix))
                        self.assertTrue(is_fully_symbolic(result_v.matrix))

    def test_relu(self):
        """Test ReLU activation for both modes."""
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(str(mode.name)):
                with matrix_mode(mode):
                    matrix = Matrix(self.values)

                    result = matrix.ReLU()

                    np.testing.assert_allclose(result.numpy(), np.maximum(self.values, 0))

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))

    def test_transpose(self):
        """Test matrix transpose for both modes."""
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(str(mode.name)):
                with matrix_mode(mode):
                    matrix = Matrix(self.values)

                    result = matrix.T

                    np.testing.assert_allclose(result.numpy(), self.values.T)

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))

    def test_incompatible_operations(self):
        """Test operations between matrices of different modes."""
        numeric_matrix = Matrix(self.values, mode=MatrixMode.NUMERIC)
        symbolic_matrix = Matrix(self.values, mode=MatrixMode.SYMBOLIC)
        with self.assertRaises(ValueError):
            _ = numeric_matrix + symbolic_matrix


class TestMatrixConstructors(unittest.TestCase):

    def setUp(self):
        # Basic dimensions setup for testing
        self.dim = 3
        self.rows = 3
        self.cols = 4

    def test_eye_constructor(self):
        """Test the identity matrix constructor for both modes."""
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(f"Mode: {mode.name}"):
                with matrix_mode(mode):
                    result = Matrix.eye(self.dim, mode=mode)
                    np.testing.assert_allclose(result.numpy(), np.eye(self.dim))
                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))

    def test_zeros_constructor(self):
        """Test the zeros matrix constructor for both modes."""
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(f"Mode: {mode.name}"):
                with matrix_mode(mode):
                    result = Matrix.zeros(self.rows, self.cols, mode=mode)
                    np.testing.assert_allclose(result.numpy(), np.zeros((self.rows, self.cols)))
                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))

    def test_ones_constructor(self):
        """Test the ones matrix constructor for both modes."""
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(f"Mode: {mode.name}"):
                with matrix_mode(mode):
                    result = Matrix.ones(self.rows, self.cols, mode=mode)
                    np.testing.assert_allclose(result.numpy(), np.ones((self.rows, self.cols)))
                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))

    def test_diag_constructor(self):
        """Test the block diagonal matrix constructor for both modes."""
        matrices = [Matrix.eye(3) * (-1), Matrix.zeros(3, 1), 2, Matrix.ones(1, 2)]
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(f"Mode: {mode.name}"):
                with matrix_mode(mode):
                    result = Matrix.diag(*matrices)
                    # Check if all blocks are correctly placed
                    expected = scipy.linalg.block_diag(
                        *[
                            -np.eye(3),
                            np.zeros((3, 1)),
                            np.eye(2),
                            np.ones((1, 2)),
                        ]
                    )
                    np.testing.assert_allclose(result.numpy(), expected)
                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))

    def test_slice_constructor(self):
        """Test the slice constructor for both modes."""
        # The interface is  SliceMatrix(cls, input_dim: int, start: int, end: int, mode=None) -> Matrix:

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with self.subTest(f"Mode: {mode.name}"):
                with matrix_mode(mode):
                    result = Matrix.SliceMatrix(3, 1, 2)
                    expected = np.zeros((1, 3))
                    expected[0, 1] = 1
                    np.testing.assert_allclose(result.numpy(), expected)
                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))

                    result = Matrix.SliceMatrix(3, 0, 2)
                    expected = np.zeros((2, 3))
                    expected[0, 0] = 1
                    expected[1, 1] = 1
                    np.testing.assert_allclose(result.numpy(), expected)
                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))

                    result = Matrix.SliceMatrix(3, 0, 3)
                    expected = np.eye(3)
                    np.testing.assert_allclose(result.numpy(), expected)
                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_fully_symbolic(result.matrix))


if __name__ == "__main__":
    unittest.main()
