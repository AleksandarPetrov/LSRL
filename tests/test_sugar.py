from itertools import cycle, islice, product
import numpy as np
import sympy
import lsrl
import unittest

from lsrl.matrix_tools import Matrix, MatrixMode, matrix_mode
from lsrl.utils import is_symbolic


class TestMagicSugar(unittest.TestCase):
    """Tests the magic methods of the Variable class."""

    def setUp(self) -> None:
        self.xs = [
            np.random.rand(6) - 0.5,
            4 * (np.random.rand(3) - 0.5),
            -2 * (np.random.rand(1) - 0.5),
            2 * (np.random.rand(5) - 2),
            np.random.randint(10, size=6),
            np.linspace(0, 10, num=6),
            np.random.normal(0, 1, size=6),
            np.random.uniform(0, 1, size=6),
            np.random.exponential(1, size=6),
            np.random.binomial(10, 0.5, size=6),
        ]
        self.ys = [
            3 * np.random.rand(6) - 0.5,
            -4 * (np.random.rand(3) - 0.5),
            2 * (np.random.rand(1) - 0.5),
            12 * (np.random.rand(5) - 2),
            np.random.randint(10, size=6),
            np.linspace(0, 10, num=6),
            np.random.normal(0, 1, size=6),
            np.random.uniform(0, 1, size=6),
            np.random.exponential(1, size=6),
            np.random.binomial(10, 0.5, size=6),
        ]

    def test_addition(self):
        """Tests the addition magic method."""
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):

                    # test addition of variables with variables
                    for x, y in zip(self.xs, self.ys):
                        input = lsrl.Input(dim=1)
                        xb = lsrl.f_constant(input, x)
                        yb = lsrl.f_constant(input, y)
                        zb = lsrl.ForEach(xb + yb)

                        np.testing.assert_allclose(zb(0).numpy().squeeze(), x + y)

                    # test addition of variables with scalars
                    variables = self.xs + self.ys
                    scalars = cycle([-1.2, -1, 0, 1, 3.1415])
                    for v, s in zip(variables, scalars):
                        input = lsrl.Input(dim=1)
                        vb = lsrl.f_constant(input, v)
                        zb = lsrl.ForEach(vb + s)
                        zb2 = lsrl.ForEach(s + vb)

                        out = zb(np.array([0]))
                        out2 = zb2(np.array([0]))

                        np.testing.assert_allclose(out.numpy().squeeze(), v + s)
                        np.testing.assert_allclose(out2.numpy().squeeze(), v + s)

                        if mode == MatrixMode.SYMBOLIC:
                            self.assertTrue(is_symbolic(out.matrix))
                            self.assertTrue(is_symbolic(out2.matrix))

    def test_subtraction(self):
        """Tests the subtraction magic method."""

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):

                    # test substraction of variables with variables
                    for x, y in zip(self.xs, self.ys):
                        input = lsrl.Input(dim=1)
                        xb = lsrl.f_constant(input, x)
                        yb = lsrl.f_constant(input, y)
                        zb = lsrl.ForEach(xb - yb)

                        np.testing.assert_allclose(zb(0).numpy().squeeze(), x - y)

                    # test substraction of variables with scalars
                    variables = self.xs + self.ys
                    scalars = cycle([-1.2, -1, 0, 1, 3.1415])
                    for v, s in zip(variables, scalars):
                        input = lsrl.Input(dim=1)
                        vb = lsrl.f_constant(input, v)
                        zb = lsrl.ForEach(vb - s)
                        zb2 = lsrl.ForEach(s - vb)

                        out = zb(np.array([0]))
                        out2 = zb2(np.array([0]))

                        np.testing.assert_allclose(out.numpy().squeeze(), v - s)
                        np.testing.assert_allclose(out2.numpy().squeeze(), s - v)

                        if mode == MatrixMode.SYMBOLIC:
                            self.assertTrue(is_symbolic(out.matrix))
                            self.assertTrue(is_symbolic(out2.matrix))

    def test_multiplication(self):
        """Tests the multiplication magic method."""

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):

                    # concat the xs and ys and create a list of scalars that has the same size which are -1.2, -1, 0, 1, 3.1415 iterating
                    variables = self.xs + self.ys
                    scalars = cycle([-1.2, -1, 0, 1, 3.1415])
                    for v, s in zip(variables, scalars):
                        input = lsrl.Input(dim=1)
                        vb = lsrl.f_constant(input, v)
                        zb = lsrl.ForEach(vb * s)
                        zb2 = lsrl.ForEach(s * vb)

                        out = zb(np.array([0]))
                        out2 = zb2(np.array([0]))

                        np.testing.assert_allclose(out.numpy().squeeze(), v * s)
                        np.testing.assert_allclose(out2.numpy().squeeze(), v * s)

                        if mode == MatrixMode.SYMBOLIC:
                            self.assertTrue(is_symbolic(out.matrix))
                            self.assertTrue(is_symbolic(out2.matrix))

    def test_division(self):
        """Tests the division magic method."""

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):

                    # concat the xs and ys and create a list of scalars that has the same size which are -1.2, -1, 0, 1, 3.1415 iterating
                    variables = self.xs + self.ys
                    scalars = cycle([-1.2, -1, 0.01, 1, 3.1415])
                    for v, s in zip(variables, scalars):
                        input = lsrl.Input(dim=1)
                        vb = lsrl.f_constant(input, v)
                        zb = lsrl.ForEach(vb / s)

                        out = zb(np.array([0]))

                        np.testing.assert_allclose(out.numpy().squeeze(), v / s)

                        if mode == MatrixMode.SYMBOLIC:
                            self.assertTrue(is_symbolic(out.matrix))

    def test_slicing(self):
        """Tests the slicing magic method."""

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):
                    # create a list of random vectors
                    vectors = [
                        np.random.rand(6) - 0.5,
                        4 * (np.random.rand(3) - 0.5),
                        -2 * (np.random.rand(1) - 0.5),
                        2 * (np.random.rand(5) - 2),
                        np.random.randint(10, size=6),
                        np.linspace(0, 10, num=6),
                        np.random.normal(0, 1, size=6),
                        np.random.uniform(0, 1, size=6),
                        np.random.exponential(1, size=6),
                        np.random.binomial(10, 0.5, size=6),
                    ]

                    # slice the vectors
                    input = lsrl.Input(dim=1)

                    for v in vectors:
                        input = lsrl.Input(dim=1)
                        vb = lsrl.f_constant(input, v)

                        for i in range(len(v)):
                            zb = lsrl.ForEach(vb[i])
                            np.testing.assert_allclose(zb(0).numpy().flatten(), v[i])

                        for start in range(len(v) - 1):
                            for end in range(start + 1, len(v)):
                                zb1 = lsrl.ForEach(vb[start:end])
                                zb2 = lsrl.ForEach(vb[:end])
                                zb3 = lsrl.ForEach(vb[start:])

                                np.testing.assert_allclose(zb1(0).numpy().squeeze(), v[start:end])
                                np.testing.assert_allclose(zb2(0).numpy().squeeze(), v[:end])
                                np.testing.assert_allclose(zb3(0).numpy().squeeze(), v[start:])


class TestBasicConstructs(unittest.TestCase):

    def test_step(self):
        """The step should be 1 if the input is greater than 1/scale, 0 otherwise."""

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode.name):

                    with self.subTest("scalar input"):
                        epsilon = 1e-5
                        for scale in [0.1, 1, 10, 100, 1000]:
                            input = lsrl.Input(dim=1)
                            step = lsrl.ForEach(lsrl.f_step(input, scale))
                            np.testing.assert_allclose(step(0).numpy().squeeze(), 0)
                            np.testing.assert_allclose(step(1 / scale).numpy().squeeze(), 1)
                            np.testing.assert_allclose(step(1 / scale + epsilon).numpy().squeeze(), 1)

                    # test with vector inputs
                    with self.subTest("vector input"):
                        vectors = [
                            [9.1, -0.2, 13, 4.2, 0, 12.4],
                            [9.1],
                            [0, 0.0001, 0.001, 0.01, 0.1, 1],
                        ]
                        for v in vectors:
                            scale = 100000
                            input = lsrl.Input(dim=len(v))
                            step = lsrl.ForEach(lsrl.f_step(input, scale=scale))
                            np.testing.assert_allclose(
                                step(np.array(v)).numpy().squeeze(),
                                np.array([1 if x > 1 / scale else 0 for x in v]),
                            )

    def test_bump(self):
        """
        The bump should be 1 if the input is between lb and ub, 0 otherwise.
        (Approximately as we are using ReLUs underneath)
        """

        lbs = [-4, -1.3, -0.01, 0, 0.01, 1.3, 10]
        ubs = [-1.4, -0.01, 0, 0.01, 1.3, 10, 100]
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(f"scalar input, {mode.name}"):
                    for lb, ub in zip(lbs, ubs):
                        if lb < ub:
                            input = lsrl.Input(dim=1)
                            lb_, ub_ = lsrl.f_constant(input, [lb]), lsrl.f_constant(input, [ub])
                            bump = lsrl.ForEach(lsrl.f_bump(input, lb_, ub_, scale=1000))
                            np.testing.assert_allclose(
                                bump(lb).numpy().squeeze(), 0, atol=1e-7
                            )  # because of the approximation
                            np.testing.assert_allclose(bump(ub).numpy().squeeze(), 1, atol=1e-7)
                            np.testing.assert_allclose(bump(lb + 0.5 * (ub - lb)).numpy().squeeze(), 1, atol=1e-7)
                            np.testing.assert_allclose(bump(lb - 1).numpy().squeeze(), 0, atol=1e-7)
                            np.testing.assert_allclose(bump(ub + 1).numpy().squeeze(), 0, atol=1e-7)

                # test with vector inputs
                with self.subTest(f"vector input, {mode.name}"):
                    lbs = [-4, -1.3, -0.01, 0, 0.01, 1.3, 10]
                    ubs = [-1.4, -0.01, 0, 0.01, 1.3, 1.0, 100]
                    vals = [-2, 1, -0.0005, 0.02, 0.31415, -1.1, 11]
                    input = lsrl.Input(dim=len(vals))
                    lb_, ub_ = lsrl.f_constant(input, lbs), lsrl.f_constant(input, ubs)
                    bump = lsrl.ForEach(lsrl.f_bump(input, lb_, ub_, scale=1000))
                    np.testing.assert_allclose(
                        bump(np.array(vals)).numpy().squeeze(),
                        np.array([1 if lb < x < ub else 0 for lb, ub, x in zip(lbs, ubs, vals)]),
                        atol=1e-7,
                    )


class TestLogicConstructs(unittest.TestCase):

    def test_equality(self):
        """The equality should be 1 if the two inputs are equal, 0 otherwise."""
        epsilon = 1e-5
        scale = 100000

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(f"scalar input, {mode.name}"):
                    for v in [-2, 1, -0.0005, 0.02, 0.31415, -1.1, 11]:
                        input = lsrl.Input(dim=1)
                        eq = lsrl.ForEach(lsrl.f_eq(input, input, scale))
                        neq = lsrl.ForEach(lsrl.f_neq(input, input, scale))
                        np.testing.assert_allclose(eq(v).numpy().squeeze(), 1)
                        np.testing.assert_allclose(neq(v).numpy().squeeze(), 0)

                # test with vector inputs
                with self.subTest(f"vector input, {mode.name}"):
                    vectors = [
                        np.array(v)
                        for v in [
                            [9.1, -0.2, 13, 13, 4.2, 0, 12.4],
                            [9.1],
                            [0, 0.0001, 0.0001, 0.001, 0.01, 0.1, 1],
                        ]
                    ]
                    for v in vectors:
                        input = lsrl.Input(dim=len(v))
                        eq = lsrl.ForEach(lsrl.f_eq(input, input, scale))
                        neq = lsrl.ForEach(lsrl.f_neq(input, input, scale))

                        np.testing.assert_allclose(eq(v).numpy().squeeze(), np.ones(len(v)))
                        np.testing.assert_allclose(neq(v).numpy().squeeze(), np.zeros(len(v)))

                        # TEST THAT x!=y:
                        x = lsrl.f_constant(input, v)
                        rolled_v = np.roll(v, 1)
                        y = lsrl.f_constant(input, rolled_v)
                        eq = lsrl.ForEach(lsrl.f_eq(x, y, scale))
                        neq = lsrl.ForEach(lsrl.f_neq(x, y, scale))

                        np.testing.assert_allclose(
                            eq(v).numpy().squeeze(), np.array(v == rolled_v, dtype=float), atol=1e-4
                        )
                        np.testing.assert_allclose(
                            neq(v).numpy().squeeze(), np.array(v != rolled_v, dtype=float), atol=1e-4
                        )

    def test_smaller_larger(self):
        """The smaller should be 1 if the first input is smaller than the second, 0 otherwise."""
        epsilon = 1e-5
        scale = 100000

        # scalar input
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(f"scalar input, {mode.name}"):
                    for x, y in zip(
                        [-2, 1, -0.0005, 0.02, 0.31415, -1.1, 11],
                        [3, 1.1, 0.0005, 0.03, 0.31416, -1.0, 12],
                    ):
                        input = lsrl.Input(dim=1)
                        smaller = lsrl.ForEach(
                            lsrl.f_smaller(lsrl.f_constant(input, x), lsrl.f_constant(input, y), scale)
                        )
                        larger = lsrl.ForEach(
                            lsrl.f_larger(lsrl.f_constant(input, x), lsrl.f_constant(input, y), scale)
                        )

                        np.testing.assert_allclose(
                            smaller(0).numpy().squeeze(), np.array([1 if x < y else 0], dtype=float)
                        )
                        np.testing.assert_allclose(
                            larger(0).numpy().squeeze(), np.array([1 if x > y else 0], dtype=float)
                        )

                # test with vector inputs
                with self.subTest(f"vector input, {mode.name}"):
                    vectors = [
                        [9.1, -0.2, 13, 4.2, 0, 12.4],
                        [9.1],
                        [0, 0.0001, 0.001, 0.01, 0.1, 1],
                    ]
                    for v in vectors:
                        input = lsrl.Input(dim=1)
                        x = lsrl.f_constant(input, v)
                        rolled_v = np.roll(v, 1)
                        y = lsrl.f_constant(input, rolled_v)
                        smaller = lsrl.ForEach(lsrl.f_smaller(x, y, scale))
                        larger = lsrl.ForEach(lsrl.f_larger(x, y, scale))

                        np.testing.assert_allclose(smaller(0).numpy().squeeze(), np.where(v < rolled_v, 1, 0))
                        np.testing.assert_allclose(larger(0).numpy().squeeze(), np.where(v > rolled_v, 1, 0))

    def test_ifeq(self):

        # this should essentially have the functionality of np.where(x==y, t_val, f_val)

        # create a list of potential values for x and y
        vectors = [
            [-1],
            [0],
            [1],
            [-2, 2],
            [1, -1],
            [-1, -1],
            [1, 1],
            [-1, 1],
            [0, 0, 0, 0, 0],
            [1, 0, -1, 0, 1],
            [-1, -1, 0, 1, 1],
        ]

        lengths = set(len(i) for i in vectors)

        # iterate through all pairs
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                for length in lengths:
                    vectors_of_len = [v for v in vectors if len(v) == length]
                    with self.subTest(length=length, mode=mode.name):
                        for x, y, t_val, f_val in product(vectors_of_len, repeat=4):
                            # raise RuntimeError(f"Failed case: x={x}, y={y}, t_val={t_val}, f_val={f_val}")
                            input = lsrl.Input(dim=1)
                            x_, y_, t_val_, f_val_ = (
                                lsrl.f_constant(input, x),
                                lsrl.f_constant(input, y),
                                lsrl.f_constant(input, t_val),
                                lsrl.f_constant(input, f_val),
                            )
                            ifeq = lsrl.ForEach(lsrl.f_ifeq(x_, y_, t_val_, f_val_, scale=10000))
                            np.testing.assert_allclose(
                                ifeq(0).numpy().squeeze(),
                                np.where(np.array(x) == np.array(y), np.array(t_val), np.array(f_val)),
                                atol=1e-4,
                                err_msg=f"Failed case: x={x}, y={y}, t_val={t_val}, f_val={f_val}",
                            )

    def test_ifelse_mul(self):
        values = [-1, 0.1, -0.1, 1, 10, -3]

        # iterate through all pairs
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(mode=mode.name):
                    for x, y, t_val, f_val in islice(product(values, repeat=4), 100):
                        input = lsrl.Input(dim=1)
                        x_, y_, t_val_, f_val_ = (
                            lsrl.f_constant(input, x),
                            lsrl.f_constant(input, y),
                            lsrl.f_constant(input, t_val),
                            lsrl.f_constant(input, f_val),
                        )

                        eq = lsrl.ForEach(lsrl.f_ifelse_mul(lsrl.f_eq(x_, y_), t_val_, f_val_))
                        smaller = lsrl.ForEach(lsrl.f_ifelse_mul(lsrl.f_smaller(x_, y_), t_val_, f_val_))
                        larger = lsrl.ForEach(lsrl.f_ifelse_mul(lsrl.f_larger(x_, y_), t_val_, f_val_))

                        np.testing.assert_allclose(eq(0).numpy().squeeze(), np.where(x == y, t_val, f_val), atol=1e-4)
                        np.testing.assert_allclose(
                            smaller(0).numpy().squeeze(), np.where(x < y, t_val, f_val), atol=1e-4
                        )
                        np.testing.assert_allclose(
                            larger(0).numpy().squeeze(), np.where(x > y, t_val, f_val), atol=1e-4
                        )

    def test_and(self):
        """The and should be 1 if both inputs are 1, 0 otherwise."""
        epsilon = 1e-5
        scale = 100000

        # scalar input
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(f"scalar input, {mode.name}"):
                    for x, y in zip([0, 1, 0, 1], [0, 1, 1, 0]):
                        input = lsrl.Input(dim=1)
                        and_ = lsrl.ForEach(lsrl.f_and(lsrl.f_constant(input, x), lsrl.f_constant(input, y), scale))
                        for numeric in [True, False]:
                            np.testing.assert_allclose(
                                and_(0).numpy().squeeze(), np.array([1 if x and y else 0], dtype=float)
                            )

                # test with vector inputs
                with self.subTest(f"vector input, {mode.name}"):
                    vectors = [
                        [0, 1, 1, 0],
                        [1],
                        [0, 0, 0, 0, 0],
                    ]
                    for v in vectors:
                        input = lsrl.Input(dim=1)
                        x = lsrl.f_constant(input, v)
                        rolled_v = np.roll(v, 1)
                        y = lsrl.f_constant(input, rolled_v)
                        and_ = lsrl.ForEach(lsrl.f_and(x, y, scale))
                        np.testing.assert_allclose(
                            and_(0).numpy().squeeze(),
                            np.array([1 if x and y else 0 for x, y in zip(v, rolled_v)], dtype=float),
                        )

    def test_or(self):
        """The or should be 1 if at least one of the inputs is 1, 0 otherwise."""
        epsilon = 1e-5
        scale = 100000

        # scalar input
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(f"scalar input, {mode.name}"):
                    for x, y in zip([0, 1, 0, 1], [0, 1, 1, 0]):
                        input = lsrl.Input(dim=1)
                        or_ = lsrl.ForEach(lsrl.f_or(lsrl.f_constant(input, x), lsrl.f_constant(input, y), scale))
                        for numeric in [True, False]:
                            np.testing.assert_allclose(
                                or_(0).numpy().squeeze(), np.array([1 if x or y else 0], dtype=float)
                            )

                # test with vector inputs
                with self.subTest(f"vector input, {mode.name}"):
                    vectors = [
                        [0, 1, 1, 0],
                        [1],
                        [0, 0, 0, 0, 0],
                    ]
                    for v in vectors:
                        input = lsrl.Input(dim=1)
                        x = lsrl.f_constant(input, v)
                        rolled_v = np.roll(v, 1)
                        y = lsrl.f_constant(input, rolled_v)
                        or_ = lsrl.ForEach(lsrl.f_or(x, y, scale))
                        np.testing.assert_allclose(
                            or_(0).numpy().squeeze(),
                            np.array([1 if x or y else 0 for x, y in zip(v, rolled_v)], dtype=float),
                        )

    def test_not(self):
        """The not should be 1 if the input is 0, 0 otherwise."""
        # scalar input
        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                with self.subTest(f"scalar input, {mode.name}"):
                    for x in [0, 1, 0, 1]:
                        input = lsrl.Input(dim=1)
                        not_ = lsrl.ForEach(lsrl.f_not(lsrl.f_constant(input, x)))
                        for numeric in [True, False]:
                            np.testing.assert_allclose(
                                not_(0).numpy().squeeze(), np.array([int(not bool(x))], dtype=float)
                            )
                with self.subTest(f"vector input, {mode.name}"):
                    vectors = [
                        [0, 1, 1, 0],
                        [1],
                        [0, 0, 1, 0, 0],
                    ]
                    for v in vectors:
                        input = lsrl.Input(dim=1)
                        x = lsrl.f_constant(input, v)
                        not_ = lsrl.ForEach(lsrl.f_not(x))
                        for numeric in [True, False]:
                            np.testing.assert_allclose(
                                not_(0).numpy().squeeze(), np.array([int(not bool(x)) for x in v], dtype=float)
                            )


class TestModuloCounter(unittest.TestCase):

    def test_modulo_counter(self):
        n_samples = 50
        input = lsrl.Input(dim=1)

        for mode in [MatrixMode.NUMERIC, MatrixMode.SYMBOLIC]:
            with matrix_mode(mode):
                for divisor in [1, 2, 3, 5, 20]:
                    with self.subTest(divisor=divisor, mode=mode.name):
                        counter = lsrl.f_modulo_counter(input, divisor)
                        loop = lsrl.ForEach(counter)

                        output = loop(Matrix.zeros(1, n_samples))
                        np.testing.assert_allclose(
                            output.numpy().squeeze(), list(islice(cycle(range(divisor)), n_samples))
                        )


if __name__ == "__main__":
    unittest.main()
