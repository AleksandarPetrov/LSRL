import numpy as np
import sympy
import lsrl
import unittest
import os
import tempfile
from parameterized import parameterized_class

from lsrl.matrix_tools import Matrix, MatrixMode, matrix_mode
from lsrl.utils import is_symbolic


def save_simplification_history_plots(simplification_history, prefix, dir_name="test_results"):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for f in os.listdir(dir_name):
        if f.startswith(prefix) and f.endswith(".png"):
            os.remove(os.path.join(dir_name, f))

    for i, (kind, simplification) in enumerate(simplification_history):
        lsrl.utils.plot_and_save_graph(simplification, f"{dir_name}/{prefix}_{i}_{kind}.png")


class IndividualBlockConstruction(unittest.TestCase):

    def test_linear(self):

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):
                    input = lsrl.Input(dim=4)
                    A = np.array(
                        [
                            [0.707, -0.707, 0, 0],
                            [0.707, 0.707, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ]
                    )
                    b = np.array([-0.1, 0.1, 1e-5, -1e-5])

                    # should fail with ValueError as we only can take Matrices as input
                    with self.assertRaises(ValueError):
                        _ = lsrl.Linear(
                            input=input,
                            A=A,
                            b=b,
                        )

                    block = lsrl.Linear(
                        input=input,
                        A=Matrix(A),
                        b=Matrix(b[:, None]),
                    )

                    test_in = np.random.random(4) - 0.5
                    test_out = block(test_in)

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_symbolic(block.A.matrix))
                        self.assertTrue(is_symbolic(block.b.matrix))
                        self.assertTrue(is_symbolic(test_out.matrix))

                    # check that the result is close to the expected one
                    np.testing.assert_allclose(np.dot(A, test_in) + b, test_out.numpy().flatten(), atol=1e-5)

    def test_slice(self):

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):

                    input = lsrl.Input(dim=4)

                    block1 = lsrl.Slice(input, 0, 1)
                    block2 = lsrl.Slice(input, 1, 3)

                    test_in = np.random.random(4) - 0.5
                    test_out = block1(sympy.ImmutableDenseMatrix(test_in))
                    # check that the result is close to the expected one
                    np.testing.assert_allclose(test_in[0], test_out.numpy().flatten(), atol=1e-5)

                    test_out2 = block2(sympy.ImmutableDenseMatrix(test_in))
                    # check that the result is close to the expected one
                    np.testing.assert_allclose(test_in[1:3], test_out2.numpy().flatten(), atol=1e-5)

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_symbolic(block1.A.matrix))
                        self.assertTrue(is_symbolic(block1.b.matrix))
                        self.assertTrue(is_symbolic(block2.A.matrix))
                        self.assertTrue(is_symbolic(block2.b.matrix))
                        self.assertTrue(is_symbolic(test_out.matrix))
                        self.assertTrue(is_symbolic(test_out2.matrix))

    def test_concat(self):

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):
                    input = lsrl.Input(dim=4)

                    block1 = lsrl.Concat([input, input])
                    block2 = lsrl.Concat([block1, input])

                    test_in = np.random.random(4) - 0.5

                    test_out = block1(test_in, sympy.ImmutableDenseMatrix(test_in))
                    # check that the result is close to the expected one
                    np.testing.assert_allclose(
                        np.concatenate([test_in, test_in]),
                        test_out.numpy().flatten(),
                        atol=1e-5,
                    )

                    test_out2 = block2(test_out, test_in)  # check that the result is close to the expected one
                    np.testing.assert_allclose(
                        np.concatenate([test_in, test_in, test_in]),
                        test_out2.numpy().flatten(),
                        atol=1e-5,
                    )

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_symbolic(test_out.matrix))
                        self.assertTrue(is_symbolic(test_out2.matrix))

    def test_relu(self):
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):
                    input = lsrl.Input(dim=4)
                    block = lsrl.ReLU(input)

                    test_in = np.random.random(4) - 0.5

                    test_out = block(sympy.ImmutableDenseMatrix(test_in))
                    # check that the result is close to the expected one
                    np.testing.assert_allclose(np.maximum(0, test_in), test_out.numpy().flatten(), atol=1e-5)

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_symbolic(test_out.matrix))

    def test_state(self):
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):

                    input = lsrl.Input(dim=4)
                    A = np.array(
                        [
                            [0.707, -0.707, 0],
                            [0.707, 0.707, 0],
                            [0, 0, 1],
                        ]
                    )
                    B = [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1]]
                    bias = np.array([-0.1, 0.1, 1e-5])[:, None]

                    block = lsrl.LinState(
                        input=input, A=Matrix(A), B=Matrix(B), init_state=Matrix.ones(3, 1), bias=Matrix(bias)
                    )

                    test_in = np.random.random(4) - 0.5
                    block.reset()
                    test_out = block(test_in)
                    np.testing.assert_allclose(
                        np.dot(A, np.ones(3)) + np.dot(B, test_in) + bias.flatten(),
                        test_out.numpy().flatten(),
                        atol=1e-5,
                    )

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_symbolic(test_out.matrix))

    def test_multiplicative(self):
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):

                    input = lsrl.Input(dim=4)
                    mul = lsrl.Multiplicative(input)

                    test_in_dense = np.random.random(4) - 0.5
                    test_out_dense = mul(test_in_dense)

                    test_in_sparse = np.array([0, -0.1, 0, 0.1])
                    test_out_sparse = mul(test_in_sparse)

                    np.testing.assert_allclose(
                        test_in_dense[:2] * test_in_dense[2:],
                        test_out_dense.numpy().flatten(),
                        atol=1e-5,
                    )
                    np.testing.assert_allclose(
                        test_in_sparse[:2] * test_in_sparse[2:],
                        test_out_sparse.numpy().flatten(),
                        atol=1e-5,
                    )

                    if mode == MatrixMode.SYMBOLIC:
                        self.assertTrue(is_symbolic(test_out_dense.matrix))
                        self.assertTrue(is_symbolic(test_out_sparse.matrix))


class TestBasicBlocks(unittest.TestCase):

    def test_basic_construction(self):

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):

                    input = lsrl.Input(dim=2)

                    # linear block that duplicates the input
                    x = lsrl.Linear(
                        input=input,
                        A=Matrix([[1, 0], [0, 1], [1, 0], [0, 1]]),
                        b=Matrix.zeros(4, 1),
                    )
                    self.assertEqual(x.dim, 4)

                    # ReLU block
                    x = lsrl.ReLU(input=x)
                    self.assertEqual(x.dim, 4)

                    # Rotate the first two dimensions by 45 degrees, keep the other two the same
                    x = lsrl.Linear(
                        input=x,
                        A=Matrix(
                            sympy.Matrix(
                                [
                                    [sympy.sqrt(2) / sympy.S(2), -sympy.sqrt(2) / sympy.S(2), 0, 0],
                                    [sympy.sqrt(2) / sympy.S(2), sympy.sqrt(2) / sympy.S(2), 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                ]
                            )
                        ),
                        b=Matrix.zeros(4, 1),
                    )
                    self.assertEqual(x.dim, 4)

                    # ReLU block
                    x = lsrl.ReLU(input=x)
                    self.assertEqual(x.dim, 4)

                    # Slice the first two dimensions
                    xA = lsrl.Slice(input=x, start=0, end=2)
                    self.assertEqual(xA.dim, 2)
                    # bias them by adding one to each dimension
                    xA = lsrl.Linear(
                        input=xA,
                        A=Matrix.eye(2),
                        b=Matrix.ones(2, 1),
                    )

                    # Slice the last two dimensions
                    xB = lsrl.Slice(input=x, start=2, end=4)
                    self.assertEqual(xB.dim, 2)
                    # flip the two dimensions
                    xB = lsrl.Linear(
                        input=xB,
                        A=Matrix([[0, 1], [-1, 0]]),
                        b=Matrix.zeros(2, 1),
                    )

                    # Concatenate the two dimensions
                    x = lsrl.Concat(inputs=[xA, xB])
                    self.assertEqual(x.dim, 4)

                    x_mul = lsrl.Multiplicative(input=x)
                    self.assertEqual(x_mul.dim, 2)

                    # Add a linear state to count the number of inputs
                    constant1 = lsrl.Linear(
                        input=input,
                        A=Matrix.zeros(1, 2),
                        b=Matrix.ones(1, 1),
                    )
                    counter = lsrl.LinState(
                        input=constant1,
                        A=Matrix.ones(1, 1),
                        B=Matrix.ones(1, 1),
                        init_state=Matrix.zeros(1, 1),
                    )

                    # Add the counter to the output
                    x = lsrl.Concat(inputs=[x, counter, x_mul])
                    self.assertEqual(x.dim, 7)

                    lsrl.utils.plot_and_save_graph(
                        x.graph(), f"test_results/test_results_basic_construction_{mode.name}.png"
                    )

    def test_forward(self):

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):

                    input = lsrl.Input(dim=2)

                    # linear block that duplicates the input
                    x = lsrl.Linear(
                        input=input,
                        A=Matrix([[1, 0], [0, 1], [1, 0], [0, 1]]),
                        b=Matrix.zeros(4, 1),
                    )
                    x = lsrl.ReLU(input=x)

                    # Slice the first two dimensions
                    xA = lsrl.Slice(input=x, start=0, end=2)
                    self.assertEqual(xA.dim, 2)
                    # bias them by adding one to each dimension
                    xA = lsrl.Linear(input=xA, A=Matrix.eye(2), b=Matrix.ones(2, 1))  # Slice the last two dimensions
                    xB = lsrl.Slice(input=x, start=2, end=4)
                    self.assertEqual(xB.dim, 2)
                    # flip the two dimensions
                    xB = lsrl.Linear(input=xB, A=Matrix([[0, 1], [1, 0]]), b=Matrix.zeros(2, 1))

                    # Concatenate the two dimensions
                    x = lsrl.Concat(inputs=[xA, xB])

                    # Add a linear state to count the number of inputs
                    constant1 = lsrl.Linear(input=input, A=Matrix.zeros(1, 2), b=Matrix.ones(1, 1))
                    counter = lsrl.LinState(
                        input=constant1,
                        A=Matrix.ones(1, 1),
                        B=Matrix.ones(1, 1),
                        init_state=Matrix.zeros(1, 1),
                    )

                    mul = lsrl.Multiplicative(input)

                    # def counter_aft_hook(s, out, x):
                    #     print(f"Counter input: {x}, Counter output: {out}, Counter state: {s.state}")
                    # counter.hooks["after_forward"].append(counter_aft_hook)
                    # def constant_hook(s, out, x):
                    #     print(f"Constant input: {x}, Constant output: {out}")
                    # constant1.hooks["after_forward"].append(constant_hook)

                    # Add the counter to the output
                    x = lsrl.Concat(inputs=[counter, x, mul])

                    # construct the loop
                    loop = lsrl.ForEach(x)

                    # generate test data
                    data_in = np.array(
                        [
                            [3, 6],
                            [4, 2],
                            [7, 12],
                            [2, 4],
                        ]
                    )
                    expected_out = np.array(
                        [
                            [1, 4, 7, 6, 3, 3 * 6],
                            [2, 5, 3, 2, 4, 4 * 2],
                            [3, 8, 13, 12, 7, 7 * 12],
                            [4, 3, 5, 4, 2, 2 * 4],
                        ]
                    )

                    # check the forward pass
                    generated = loop(data_in.T)
                    np.testing.assert_allclose(expected_out.T, generated.numpy(), atol=1e-5)


@parameterized_class([{"mode": mode, "name": {mode.name}} for mode in MatrixMode])
class TestSimplification(unittest.TestCase):

    mode: MatrixMode

    def setUp(self) -> None:
        self._mode_manager = matrix_mode(self.mode)
        self._mode_manager.__enter__()

    def tearDown(self):
        self._mode_manager.__exit__(None, None, None)

    def perform_the_test(self, output, plots_name: str, expected_final_size: int = None):
        loop = lsrl.ForEach(output, keep_graph_history=True, verbose=True)

        # some random data in
        data_in = np.random.randn(20, loop.input.dim)
        generated = loop(data_in.T)

        if self.mode == MatrixMode.SYMBOLIC:
            self.assertTrue(is_symbolic(generated.matrix))

        generated = generated.numpy()

        # count the number of nodes before simplification
        n_nodes_initially = loop.graph().number_of_nodes()
        parameter_count_initially = loop.parameter_count

        try:
            loop.simplify(max_steps=100)
            save_simplification_history_plots(loop.simplification_history, plots_name)
        except Exception as e:
            save_simplification_history_plots(loop.simplification_history, plots_name)
            raise e

        # check that the forward pass hasn't changed
        generated_after = loop(data_in.T).numpy()
        np.testing.assert_allclose(generated, generated_after, atol=1e-5)

        print(f"Parameter count before simplification: {parameter_count_initially:,}")
        print(f"Parameter count after simplification: {loop.parameter_count:,}")

        if expected_final_size is not None:
            n_nodes_after = loop.graph().number_of_nodes()
            self.assertEqual(n_nodes_after, expected_final_size)

        # check the result is a path graph
        self.assertTrue(lsrl.utils.is_directed_path_graph(loop.graph()))

    def test_only_states(self):
        input = lsrl.Input(dim=2)
        # Add a split with only linear states to test fusing
        only_states1 = lsrl.LinState(
            input=input,
            A=Matrix.eye(4),
            B=Matrix.vstack(Matrix.eye(2), Matrix.eye(2)),
            init_state=Matrix.zeros(4, 1),
        )
        only_states2 = lsrl.LinState(
            input=input,
            A=Matrix.eye(4) * (-1),
            B=Matrix.vstack(Matrix.eye(2), Matrix.eye(2)) * sympy.Rational(1, 2),
            init_state=Matrix.ones(4, 1),
        )
        only_states3 = lsrl.LinState(
            input=input,
            A=Matrix([[0.7, 0.3], [0.3, 0.7]]),
            B=Matrix.ones(2, 2) * sympy.Rational(1, 2),
            init_state=Matrix([[0.1, 0.9]]).T,
        )
        together = lsrl.Concat([only_states1, only_states2, only_states3])
        sum_first_five_second_five = lsrl.Linear(
            input=together,
            A=Matrix.vstack(
                Matrix.hstack(Matrix.ones(5, 1), Matrix.zeros(5, 1)),
                Matrix.hstack(Matrix.zeros(5, 1), Matrix.ones(5, 1)),
            ).T,
            b=Matrix.zeros(2, 1),
        )

        self.perform_the_test(sum_first_five_second_five, f"test_only_states_{self.mode.name}", 3)

    def test_only_relus(self):
        input = lsrl.Input(dim=2)
        # Add a split with only linear states to test fusing
        only_relu1 = lsrl.ReLU(
            input=input,
        )
        only_relu2 = lsrl.ReLU(
            input=input,
        )
        only_relu3 = lsrl.ReLU(
            input=input,
        )
        together = lsrl.Concat([only_relu1, only_relu2, only_relu3])
        sum_first_five_second_five = lsrl.Linear(
            input=together,
            A=Matrix.vstack(
                Matrix.hstack(Matrix.ones(1, 3), Matrix.zeros(1, 3)),
                Matrix.hstack(Matrix.zeros(1, 3), Matrix.ones(1, 3)),
            ),
            b=Matrix.zeros(2, 1),
        )

        self.perform_the_test(sum_first_five_second_five, f"test_only_relus_{self.mode.name}", 3)

    def test_only_multiplicatives(self):
        input = lsrl.Input(dim=6)
        # Add a split with only linear states to test fusing
        mul1 = lsrl.Multiplicative(
            input=input,
        )
        mul2 = lsrl.Multiplicative(
            input=input,
        )
        together = lsrl.Concat([mul1, mul2])
        sum = lsrl.Linear(
            input=together,
            A=Matrix.vstack(
                Matrix.hstack(Matrix.ones(1, 3), Matrix.zeros(1, 3)),
                Matrix.hstack(Matrix.zeros(1, 3), Matrix.ones(1, 3)),
            ),
            b=Matrix.zeros(2, 1),
        )

        self.perform_the_test(sum, f"test_only_multiplicatives_{self.mode.name}", 3)

    def test_only_concats_simple1(self):

        input = lsrl.Input(dim=2)
        self.perform_the_test(lsrl.Concat([input, input]), f"test_only_concats_simple1_{self.mode.name}", 2)

    def test_only_concats_simple2(self):

        input = lsrl.Input(dim=2)
        c1 = lsrl.Concat([input])
        c2 = lsrl.Concat([input, input])
        c3 = lsrl.Concat([c1, c2])

        self.perform_the_test(c3, f"test_only_concats_simple2_{self.mode.name}", 2)

    def test_only_concats_complex(self):

        input = lsrl.Input(dim=2)
        c1 = lsrl.Concat([input])
        c2 = lsrl.Concat([input, input])
        c3 = lsrl.Concat([c1, c2])
        c4 = lsrl.Concat([c2, input, c1])
        c5 = lsrl.Concat([c3, c4])
        self.perform_the_test(c5, f"test_only_concats_complex_{self.mode.name}", 2)

    def test_only_linear(self):
        input = lsrl.Input(dim=2)
        op1 = lsrl.Linear(
            input,
            A=Matrix.eye(2),
            b=Matrix([[-5, -5]]).T,
        )

        op2 = lsrl.Linear(
            input,
            A=Matrix.eye(2),
            b=Matrix([[-5, 5]]).T,
        )

        self.perform_the_test(lsrl.Concat([op1, op2]), f"test_only_linear_{self.mode.name}", 2)

    def test_only_slices(self):
        input = lsrl.Input(dim=2)
        s1 = input[0:1]
        s2 = input[1:2]
        s3 = input[0:2]
        s4 = input[0]
        op1 = lsrl.ReLU(s1)
        op2 = lsrl.Linear(s2, A=Matrix([[1]]), b=Matrix([[-1]]))
        op3 = lsrl.LinState(
            input=s3,
            A=Matrix([[0.7, 0.3], [0.3, 0.7]]),
            B=Matrix([[0.5, 0.5], [0.5, 0.5]]),
            init_state=Matrix([[0.0, 0.9]]).T,
        )
        op4 = lsrl.Concat([op1, op2, op3, s4])
        self.perform_the_test(op4, f"test_only_slices_{self.mode.name}", 5)

    def test_some_states1(self):
        input = lsrl.Input(dim=2)
        lin = lsrl.Linear(input, A=Matrix([[1, 0], [0, 1]]), b=Matrix.zeros(2, 1))
        state = lsrl.LinState(
            init_state=Matrix([[0.1], [0.9]]),
            input=input,
            A=Matrix([[0.7, 0.3], [0.3, 0.7]]),
            B=Matrix([[0.5, 0.5], [0.5, 0.5]]),
        )

        self.perform_the_test(lsrl.Concat([lin, state]), f"test_some_states1_{self.mode.name}", 3)

    def test_some_states2(self):

        # 2d rotation matrix
        make_rotation_matrix = lambda angle: Matrix(
            sympy.Matrix(
                [
                    [sympy.cos(sympy.rad(angle)), -sympy.sin(sympy.rad(angle))],
                    [sympy.sin(sympy.rad(angle)), sympy.cos(sympy.rad(angle))],
                ]
            )
        )

        input = lsrl.Input(dim=2)

        rot1 = lsrl.Linear(
            input=input,
            A=make_rotation_matrix(12),
            b=Matrix.zeros(2, 1),
        )
        rot2 = lsrl.Linear(
            input=input,
            A=make_rotation_matrix(87),
            b=Matrix.zeros(2, 1),
        )

        state = lsrl.LinState(
            init_state=Matrix([[0.1], [0.9]]),
            input=input,
            A=Matrix([[0.7, 0.3], [0.3, 0.7]]),
            B=Matrix([[0.5, 0.5], [0.5, 0.5]]),
        )

        together = lsrl.Concat([rot1, rot2, state])

        sum_first_five_second_five = lsrl.Linear(
            input=together,
            A=Matrix.vstack(
                Matrix.hstack(Matrix.ones(1, 3), Matrix.zeros(1, 3)),
                Matrix.hstack(Matrix.zeros(1, 3), Matrix.ones(1, 3)),
            ),
            b=Matrix.zeros(2, 1),
        )
        self.perform_the_test(sum_first_five_second_five, f"test_some_states2_{self.mode.name}", 3)

    def test_relu_and_linear(self):

        # 2d rotation matrix
        make_rotation_matrix = lambda angle: Matrix(
            sympy.Matrix(
                [
                    [sympy.cos(sympy.rad(angle)), -sympy.sin(sympy.rad(angle))],
                    [sympy.sin(sympy.rad(angle)), sympy.cos(sympy.rad(angle))],
                ]
            )
        )

        input = lsrl.Input(dim=2)

        rot1 = lsrl.Linear(
            input=input,
            A=make_rotation_matrix(12),
            b=Matrix.zeros(2, 1),
        )
        rot2 = lsrl.Linear(
            input=lsrl.ReLU(input),
            A=make_rotation_matrix(87),
            b=Matrix.zeros(2, 1),
        )

        rot3 = lsrl.ReLU(
            lsrl.Linear(
                input=input,
                A=make_rotation_matrix(12),
                b=Matrix.zeros(2, 1),
            )
        )

        together = lsrl.Concat([rot1, rot2, rot3])

        sum_first_five_second_five = lsrl.Linear(
            input=together,
            A=Matrix.vstack(
                Matrix.hstack(Matrix.ones(1, 3), Matrix.zeros(1, 3)),
                Matrix.hstack(Matrix.zeros(1, 3), Matrix.ones(1, 3)),
            ),
            b=Matrix.zeros(2, 1),
        )
        self.perform_the_test(sum_first_five_second_five, f"test_relu_and_linear_{self.mode.name}", 4)
        loop = lsrl.ForEach(sum_first_five_second_five)


class TestComputationGraphSimplification(unittest.TestCase):

    def test_all(self):

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                input = lsrl.Input(dim=2)

                z = lsrl.ReLU(input)
                y = lsrl.Linear(z, A=Matrix.eye(2), b=(-5) * Matrix.ones(2, 1))
                y = lsrl.ReLU(y)

                # linear block that duplicates the input
                x = lsrl.Linear(
                    input=input,
                    A=Matrix.vstack(Matrix.eye(2), Matrix.eye(2)),
                    b=Matrix.zeros(4, 1),
                )
                x = lsrl.ReLU(input=x)
                x = lsrl.ReLU(input=x)

                x = lsrl.Concat(inputs=[x, y])
                x = lsrl.Linear(
                    x,
                    A=Matrix(
                        [
                            [1, 0, 0, 0, 1, 0],
                            [0, 1, 0, 0, 0, 1],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                        ]
                    ),
                    b=Matrix.zeros(4, 1),
                )

                x2 = 2 * x - 1
                mul = lsrl.Multiplicative(x)
                mul2 = lsrl.Multiplicative(x2)
                # mul = x[:2]

                # Slice the first two dimensions
                xA = lsrl.Slice(input=x, start=0, end=2)
                self.assertEqual(xA.dim, 2)
                # bias them by adding one to each dimension
                xA = lsrl.Linear(input=xA, A=Matrix.eye(2), b=Matrix.ones(2, 1))

                # Slice the last two dimensions
                xB = lsrl.Slice(input=x, start=2, end=4)
                self.assertEqual(xB.dim, 2)
                # flip the two dimensions
                xB = lsrl.Linear(input=xB, A=Matrix([[0, 1], [1, 0]]), b=Matrix.zeros(2, 1))

                # Concatenate the two dimensions
                x = lsrl.Concat(inputs=[xA, xB])

                # Add a linear state to count the number of inputs
                constant1 = lsrl.Linear(input=z, A=Matrix([[0, 0]]), b=Matrix([[1]]))
                counter = lsrl.LinState(
                    input=constant1,
                    A=Matrix([[1]]),
                    B=Matrix([[1]]),
                    init_state=Matrix([[0]]),
                )

                # Add the counter to the output
                x = lsrl.Concat(inputs=[counter, x, mul + mul2])

                # construct the loop
                self.loop = lsrl.ForEach(x, keep_graph_history=True, verbose=True)

                # generate test data
                self.data_in = np.array(
                    [
                        [3, 6],
                        [4, 2],
                        [7, 12],
                        [2, 4],
                    ]
                )

                n_nodes_initially = self.loop.graph().number_of_nodes()
                generated_before_simplification = self.loop(self.data_in.T)

                with self.subTest(f"{mode.name}. 1 simplification"):
                    self.loop.simplify(max_steps=500)

                    # plot the graph after each simplificaiton step:
                    save_simplification_history_plots(self.loop.simplification_history, f"test_all_{mode.name}")

                    n_nodes_after = self.loop.graph().number_of_nodes()
                    print(f"Number of nodes before: {n_nodes_initially}, after: {n_nodes_after}")
                    self.assertLess(n_nodes_after, n_nodes_initially)

                    self.assertEqual(n_nodes_after, 11)

                    # check the result is a path graph
                    self.assertTrue(lsrl.utils.is_directed_path_graph(self.loop.graph()))

                with self.subTest(f"{mode.name}. 2 before/after consistency"):
                    # check that forward still works
                    generated_after = self.loop(self.data_in.T)
                    np.testing.assert_allclose(
                        generated_after.numpy(), generated_before_simplification.numpy(), atol=1e-10
                    )

                with self.subTest(f"{mode.name}. 3 read/write consistency"):
                    # check writing and saving

                    # save to a tmp file (use library to get a tmp filename)
                    with tempfile.NamedTemporaryFile(delete=True) as f:
                        filename = f.name
                        self.loop.save(filename)
                        loaded = lsrl.ForEach.load(filename)
                        generated_loaded = loaded(self.data_in.T)
                        np.testing.assert_allclose(
                            generated_loaded.numpy(), generated_before_simplification.numpy(), atol=1e-10
                        )

                # free up memory
                del self.loop.simplification_history


if __name__ == "__main__":
    unittest.main()
