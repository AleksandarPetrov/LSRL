from __future__ import annotations
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import scipy
import scipy.sparse
import sympy
from contextlib import contextmanager

from lsrl.utils import diag_matrices_and_eyes


class MatrixMode(Enum):
    NUMERIC = 1
    SYMBOLIC = 2


def set_global_matrix_mode(mode: MatrixMode):
    global GLOBAL_MATRIX_MODE
    GLOBAL_MATRIX_MODE = mode


@contextmanager
def matrix_mode(mode: MatrixMode):
    global GLOBAL_MATRIX_MODE
    previous_mode = globals().get("GLOBAL_MATRIX_MODE", MatrixMode.SYMBOLIC)
    GLOBAL_MATRIX_MODE = mode
    try:
        yield
    finally:
        GLOBAL_MATRIX_MODE = previous_mode


class Matrix:
    """
    Handling both numeric and symbolic matrices with the same interface.
    Should always be 2-dimensional.
    This should be considered immutable.
    """

    def __init__(
        self,
        matrix: List[List[float | int]] | np.ndarray | scipy.sparse.sparray | sympy.MatrixBase,
        mode: Optional[MatrixMode] = None,
        skip_symbolic_check=False,
    ):
        self.mode = self._get_mode(mode)

        # convert to the right mode
        if self.mode is MatrixMode.NUMERIC:
            if isinstance(matrix, list):
                matrix = scipy.sparse.csr_matrix(np.array(matrix))
            elif isinstance(matrix, np.ndarray):
                matrix = scipy.sparse.csr_matrix(matrix)
            elif isinstance(matrix, sympy.MatrixBase):
                matrix = scipy.sparse.csr_matrix(np.array(matrix, dtype=float))
            elif scipy.sparse.issparse(matrix):
                matrix = scipy.sparse.csr_array(matrix)
            else:
                raise ValueError(f"Invalid matrix type: {type(matrix)}")
        elif self.mode is MatrixMode.SYMBOLIC:
            if isinstance(matrix, list):
                matrix = sympy.SparseMatrix(matrix)
            elif scipy.sparse.issparse(matrix):
                matrix = sympy.SparseMatrix(matrix.toarray())
            elif isinstance(matrix, (sympy.MatrixBase, np.ndarray)):
                matrix = sympy.SparseMatrix(matrix)
            else:
                raise ValueError(f"Invalid matrix type: {type(matrix)}")

            # this ensures no floats as they break the exact
            if not skip_symbolic_check and len(matrix.atoms(sympy.Float)) > 0:
                matrix = sympy.SparseMatrix(sympy.nsimplify(matrix, rational=True))
        else:
            raise ValueError(f"Invalid matrix mode: {self.mode}")

        if len(matrix.shape) != 2:
            raise ValueError(f"Matrix must be 2-dimensional, got {matrix.shape}")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(f"Matrix must have at least one row and one column, got {matrix.shape}")

        if self.mode is MatrixMode.NUMERIC and not scipy.sparse.issparse(matrix):
            matrix = scipy.sparse.csr_matrix(matrix)
        elif self.mode is MatrixMode.SYMBOLIC and not isinstance(matrix, sympy.MatrixBase):
            raise ValueError(f"Symbolic matrix must be a sympy.MatrixBase. It is {type(matrix)}")

        self.matrix = matrix

    @classmethod
    def _get_mode(cls, mode: Optional[MatrixMode] = None) -> MatrixMode:
        if mode is None:
            mode = globals().get("GLOBAL_MATRIX_MODE", MatrixMode.SYMBOLIC)

        if mode not in MatrixMode:
            raise ValueError(f"Invalid matrix mode: {mode}")

        return mode

    @classmethod
    def eye(cls, dim, mode: Optional[MatrixMode] = None) -> Matrix:
        mode = cls._get_mode(mode)

        if mode is MatrixMode.NUMERIC:
            return cls(matrix=scipy.sparse.eye_array(dim), mode=mode)
        elif mode is MatrixMode.SYMBOLIC:
            return cls(matrix=sympy.eye(dim), mode=mode)
        else:
            raise ValueError(f"Invalid matrix mode: {mode}")

    @classmethod
    def zeros(cls, rows, cols, mode: Optional[MatrixMode] = None) -> Matrix:
        mode = cls._get_mode(mode)

        if mode is MatrixMode.NUMERIC:
            return cls(matrix=scipy.sparse.csr_array((rows, cols), dtype=float), mode=mode)
        elif mode is MatrixMode.SYMBOLIC:
            return cls(matrix=sympy.SparseMatrix.zeros(rows, cols), mode=mode)
        else:
            raise ValueError(f"Invalid matrix mode: {mode}")

    @classmethod
    def ones(cls, rows, cols, mode: Optional[MatrixMode] = None) -> Matrix:
        mode = cls._get_mode(mode)

        if mode is MatrixMode.NUMERIC:
            return cls(matrix=scipy.sparse.csr_array(np.ones((rows, cols))), mode=mode)
        elif mode is MatrixMode.SYMBOLIC:
            return cls(matrix=sympy.SparseMatrix.ones(rows, cols), mode=mode)
        else:
            raise ValueError(f"Invalid matrix mode: {mode}")

    @classmethod
    def diag(cls, *matrices: List[Matrix | int]) -> Matrix:

        if any([not isinstance(m, (Matrix, int)) for m in matrices]):
            raise ValueError("All matrices must be of type Matrix or int.")

        matrices_modes = set([m.mode for m in matrices if isinstance(m, Matrix)])
        if len(matrices_modes) == 0:
            mode = cls._get_mode(None)
        elif len(matrices_modes) == 1:
            mode = matrices_modes.pop()
        else:
            raise ValueError("All matrices must have the same mode.")

        if mode is MatrixMode.NUMERIC:
            return cls(
                scipy.sparse.block_diag(
                    [scipy.sparse.eye_array(m) if isinstance(m, int) else m.matrix for m in matrices]
                ),
                mode=mode,
            )
        elif mode is MatrixMode.SYMBOLIC:
            return cls(
                matrix=diag_matrices_and_eyes(
                    matrices=[sympy.SparseMatrix.eye(m) if isinstance(m, int) else m.matrix for m in matrices]
                ),
                mode=mode,
            )
        else:
            raise ValueError(f"Invalid matrix mode: {mode}")

    @classmethod
    def vstack(cls, *args) -> Matrix:
        # check that all arguments are matrices of the same mode
        if any([not isinstance(a, Matrix) for a in args]):
            raise ValueError("All arguments must be of type Matrix.")
        if any([a.mode != args[0].mode for a in args]):
            raise ValueError("All matrices must have the same mode.")

        mode = args[0].mode

        if mode is MatrixMode.NUMERIC:
            return Matrix(scipy.sparse.vstack([a.matrix for a in args]), mode=mode, skip_symbolic_check=True)
        elif mode is MatrixMode.SYMBOLIC:
            return Matrix(sympy.SparseMatrix.vstack(*[a.matrix for a in args]), mode=mode, skip_symbolic_check=True)
        else:
            raise ValueError(f"Invalid matrix mode: {mode}")

    @classmethod
    def hstack(cls, *args) -> Matrix:
        # check that all arguments are matrices of the same mode
        if any([not isinstance(a, Matrix) for a in args]):
            raise ValueError("All arguments must be of type Matrix.")
        if any([a.mode != args[0].mode for a in args]):
            raise ValueError("All matrices must have the same mode.")

        mode = args[0].mode

        if mode is MatrixMode.NUMERIC:
            return Matrix(scipy.sparse.hstack([a.matrix for a in args]), mode=mode, skip_symbolic_check=True)
        elif mode is MatrixMode.SYMBOLIC:
            return Matrix(sympy.SparseMatrix.hstack(*[a.matrix for a in args]), mode=mode, skip_symbolic_check=True)
        else:
            raise ValueError(f"Invalid matrix mode: {mode}")

    @classmethod
    def elementwise_multiplication(cls, m1: Matrix, m2: Matrix) -> Matrix:
        if not isinstance(m1, Matrix) or not isinstance(m2, Matrix):
            raise ValueError("Both arguments must be of type Matrix.")
        if m1.mode != m2.mode:
            raise ValueError("Both matrices must have the same mode.")
        if m1.shape != m2.shape:
            raise ValueError("Both matrices must have the same shape.")

        mode = m1.mode

        if mode is MatrixMode.NUMERIC:
            return Matrix(m1.matrix.multiply(m2.matrix), mode=mode, skip_symbolic_check=True)
        elif mode is MatrixMode.SYMBOLIC:
            # not sure if that works directly with the sparse representation or converts to dense first...
            return Matrix(m1.matrix.multiply_elementwise(m2.matrix), mode=mode, skip_symbolic_check=True)
        else:
            raise ValueError(f"Invalid matrix mode: {mode}")

    @classmethod
    def SliceMatrix(cls, input_dim: int, start: int, end: int, mode=None) -> Matrix:
        mode = cls._get_mode(mode)

        output_dim = end - start

        if mode is MatrixMode.NUMERIC:
            # make scipy sparse:
            data = np.ones(output_dim)
            row_indices = np.arange(output_dim, dtype=int)
            col_indices = np.arange(start, end, dtype=int)
            return Matrix(
                scipy.sparse.csr_matrix((data, (row_indices, col_indices)), shape=(output_dim, input_dim)),
                mode=mode,
                skip_symbolic_check=True,
            )
            # return Matrix(None, mode=mode, skip_symbolic_check=True)
        elif mode is MatrixMode.SYMBOLIC:
            return Matrix(
                sympy.SparseMatrix(
                    output_dim, input_dim, {(i, i + start): sympy.core.numbers.One() for i in range(output_dim)}
                ),
                mode=mode,
                skip_symbolic_check=True,
            )
        else:
            raise ValueError(f"Invalid matrix mode: {mode}")

    def ReLU(self) -> Matrix:
        if self.mode is MatrixMode.NUMERIC:
            return Matrix(self.matrix.maximum(0), mode=self.mode, skip_symbolic_check=True)
        elif self.mode is MatrixMode.SYMBOLIC:
            return Matrix(self.matrix.applyfunc(lambda t: sympy.Max(t, 0)), mode=self.mode, skip_symbolic_check=True)
        else:
            raise ValueError(f"Invalid matrix mode: {self.mode}")

    def _check_operands_are_compatible(self: Matrix, other: Matrix | int | float | sympy.Basic):

        # if both are matrices but in different modes, we don't know what to do
        if isinstance(other, Matrix) and other.mode != self.mode:
            raise ValueError(f"Cannot operate on matrices with different modes: {self.mode} and {other.mode}")

        # if self is Matrix in numeric and other is sympy object convert
        # other to numeric
        if self.mode is MatrixMode.NUMERIC and isinstance(other, (sympy.Basic)):
            other = np.array(other, dtype=float)

        # same if self is in symbolic and the other is not
        if self.mode is MatrixMode.SYMBOLIC and isinstance(other, (int, float)):
            other = sympy.S(other)
            if len(other.atoms(sympy.Float)) > 0:
                other = sympy.nsimplify(other, rational=True)

        return other

    def __add__(self, other: Matrix | int | float | sympy.Basic) -> Matrix:
        other = self._check_operands_are_compatible(other)
        if self.mode is MatrixMode.NUMERIC:
            return Matrix(
                self.matrix.toarray() + (other.matrix.toarray() if isinstance(other, Matrix) else other),
                mode=self.mode,
                skip_symbolic_check=True,
            )
        elif self.mode is MatrixMode.SYMBOLIC:
            return Matrix(
                self.matrix + (other.matrix if isinstance(other, Matrix) else sympy.ones(*self.matrix.shape) * other),
                mode=self.mode,
                skip_symbolic_check=True,
            )
        else:
            raise RuntimeError

    def __radd__(self, other: Matrix | int | float | sympy.Basic | np.ndarray) -> Matrix:
        return self.__add__(other)

    def __sub__(self, other: Matrix | int | float | sympy.Basic | np.ndarray):
        other = self._check_operands_are_compatible(other)
        if self.mode is MatrixMode.NUMERIC:
            return Matrix(
                self.matrix.toarray() - (other.matrix.toarray() if isinstance(other, Matrix) else other),
                mode=self.mode,
                skip_symbolic_check=True,
            )
        elif self.mode is MatrixMode.SYMBOLIC:
            return Matrix(
                self.matrix - (other.matrix if isinstance(other, Matrix) else sympy.ones(*self.matrix.shape) * other),
                mode=self.mode,
                skip_symbolic_check=True,
            )
        else:
            raise RuntimeError

    def __rsub__(self, other: Matrix | int | float | sympy.Basic | np.ndarray) -> Matrix:
        other = self._check_operands_are_compatible(other)
        if self.mode is MatrixMode.NUMERIC:
            return Matrix(
                (other.matrix.toarray() if isinstance(other, Matrix) else other) - self.matrix.toarray(),
                mode=self.mode,
                skip_symbolic_check=True,
            )
        elif self.mode is MatrixMode.SYMBOLIC:
            return Matrix(
                (other.matrix if isinstance(other, Matrix) else sympy.ones(*self.matrix.shape) * other) - self.matrix,
                mode=self.mode,
                skip_symbolic_check=True,
            )
        else:
            raise RuntimeError

    def __mul__(self, other: int | float | sympy.Basic) -> Matrix:
        other = self._check_operands_are_compatible(other)

        if isinstance(other, Matrix):
            raise ValueError(
                f"Cannot multiply {self} with {other}. Need to be a scalar. Perhaps use @ for matrix multiplication."
            )

        if self.mode is MatrixMode.NUMERIC:
            return Matrix(self.matrix * other, mode=self.mode, skip_symbolic_check=True)
        elif self.mode is MatrixMode.SYMBOLIC:
            return Matrix(self.matrix * other, mode=self.mode, skip_symbolic_check=True)
        else:
            raise RuntimeError

    def __rmul__(self, other: int | float | sympy.Basic | np.ndarray) -> Matrix:
        return self.__mul__(other)

    def __matmul__(self, other: Matrix) -> Matrix:
        other = self._check_operands_are_compatible(other)
        if not isinstance(other, Matrix):
            raise ValueError(f"Cannot multiply {self} with {other}. Need to be a Matrix.")

        if self.mode is MatrixMode.NUMERIC:
            return Matrix(self.matrix @ other.matrix, mode=self.mode, skip_symbolic_check=True)
        elif self.mode is MatrixMode.SYMBOLIC:
            return Matrix(self.matrix * other.matrix, mode=self.mode, skip_symbolic_check=True)
        else:
            raise ValueError(f"Invalid matrix mode: {self.mode}")

    def __getitem__(self, key: int | slice | Tuple[slice, slice]) -> Matrix:

        if isinstance(key, int):
            if self.cols > 1:
                raise IndexError(f"Cannot get a single element from a matrix with multiple columns.")

            if self.mode is MatrixMode.NUMERIC:
                return Matrix(self.matrix[key], mode=self.mode, skip_symbolic_check=True)
            elif self.mode is MatrixMode.SYMBOLIC:
                return Matrix([[self.matrix[key]]], mode=self.mode, skip_symbolic_check=True)
            else:
                raise ValueError(f"Invalid matrix mode: {self.mode}")

        elif isinstance(key, slice):
            # we do not support steps
            if key.step is not None:
                raise IndexError("Matrix does not support steps in slicing")
            if self.cols > 1:
                raise IndexError(
                    f"Cannot get a single element from a matrix with multiple columns. Use slicing like (row, col)"
                )

            if self.mode is MatrixMode.NUMERIC:
                return Matrix(self.matrix[key], mode=self.mode, skip_symbolic_check=True)
            elif self.mode is MatrixMode.SYMBOLIC:
                return Matrix(
                    [
                        [el]
                        for el in self.matrix[
                            key.start if key.start is not None else 0 : key.stop if key.stop is not None else self.rows
                        ]
                    ],
                    mode=self.mode,
                    skip_symbolic_check=True,
                )
            else:
                raise ValueError(f"Invalid matrix mode: {self.mode}")
        elif isinstance(key, tuple):
            # do this being senstivie to the shapes and make sure that you construct a new matrix that is 2D regardless of the slices
            if len(key) != 2:
                raise IndexError("Matrix supports only 2D slicing")

            # if we are slciing a single element
            if isinstance(key[0], int) and isinstance(key[1], int):
                if self.mode is MatrixMode.NUMERIC:
                    return Matrix([[self.matrix[key]]], mode=self.mode, skip_symbolic_check=True)
                elif self.mode is MatrixMode.SYMBOLIC:
                    return Matrix([[self.matrix[key]]], mode=self.mode, skip_symbolic_check=True)
                else:
                    raise ValueError(f"Invalid matrix mode: {self.mode}")
            # if we are slicing a row
            elif isinstance(key[0], int) and isinstance(key[1], slice):
                if self.mode is MatrixMode.NUMERIC:
                    return Matrix(self.matrix[[key[0]], key[1]], mode=self.mode, skip_symbolic_check=True)
                elif self.mode is MatrixMode.SYMBOLIC:
                    return Matrix(self.matrix[key[0], key[1]], mode=self.mode, skip_symbolic_check=True)
                else:
                    raise ValueError(f"Invalid matrix mode: {self.mode}")
            # if we are slicing a column
            elif isinstance(key[0], slice) and isinstance(key[1], int):
                if self.mode is MatrixMode.NUMERIC:
                    return Matrix(self.matrix[key[0], [key[1]]], mode=self.mode, skip_symbolic_check=True)
                elif self.mode is MatrixMode.SYMBOLIC:
                    return Matrix(self.matrix[key[0], key[1]], mode=self.mode, skip_symbolic_check=True)
                else:
                    raise ValueError(f"Invalid matrix mode: {self.mode}")
            # we are slicing across both dimensions
            elif isinstance(key[0], slice) and isinstance(key[1], slice):
                return Matrix(self.matrix[key[0], key[1]], mode=self.mode, skip_symbolic_check=True)
            else:
                raise IndexError(f"Invalid slicing: {key}")

            # return Matrix(self.matrix[key], mode=self.mode, skip_symbolic_check=True)
        else:
            raise IndexError(f"Invalid key type {type(key)}")

        # sliced = self.matrix.__getitem__(*args)
        # # ensure sliced is 2D
        # if not isinstance(sliced, (sympy.MatrixBase, scipy.sparse.spmatrix)):
        #     sliced = [[sliced]]
        # elif len(sliced.shape) == 1:
        #     sliced = sliced.reshape(1, -1)

        # return self(sliced, mode=self.mode)
        # return self.numpy().__getitem__(*args)

    @property
    def T(self) -> Matrix:
        return Matrix(self.matrix.T, mode=self.mode, skip_symbolic_check=True)

    def numpy(self) -> np.ndarray:
        if self.mode is MatrixMode.NUMERIC:
            return self.matrix.toarray()
        elif self.mode is MatrixMode.SYMBOLIC:
            return np.array(self.matrix, dtype=float)
        else:
            raise ValueError(f"Invalid matrix mode: {self.mode}")

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def rows(self) -> int:
        return self.matrix.shape[0]

    @property
    def cols(self) -> int:
        return self.matrix.shape[1]

    def __str__(self) -> str:
        return str(f"{self.mode.name}:\n" + str(self.matrix))

    def __repr__(self) -> str:
        return self.__str__()
