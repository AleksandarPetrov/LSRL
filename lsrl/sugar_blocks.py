from typing import Optional
import numpy as np
import sympy

from lsrl.matrix_tools import Matrix
from .basic_blocks import Multiplicative, ReLU, Linear, Concat, Variable, Slice, LinState


def f_constant(input: Variable, value: np.array) -> Variable:
    """This is a convenience function that creates a constant function block. The input is ignored and the output is a constant value."""

    if isinstance(value, (int, float)):
        value = np.array([value])
    elif isinstance(value, list):
        value = np.array(value)

    assert len(np.array(value).shape) == 1, f"Expected a 1D array, got {np.array(value).shape}"

    output_dim = len(value)

    r = Linear(input, A=Matrix.zeros(output_dim, input.dim), b=Matrix(value[:, None]), name="Const")

    if all(value >= 0):
        r.always_positive = True

    return r


def f_step(input: Variable, scale: int | float = 1000) -> Variable:
    """
    This is a convenience function that creates a step function block.
    The output is 1 if the input is greater than 0, 0 otherwise.
    We can't do it exactly so we will approximate it with ReLUs.
    The higher the scale, the better the approximation.

    Works with scalar and vector-valued inputs.
    """

    if not isinstance(scale, (int, float)):
        raise ValueError(f"Expected scale to be a number, got {type(scale)}")

    assert scale > 0, f"Expected scale to be positive, got {scale}"

    r = ReLU(scale * input) - scale * ReLU(input - (sympy.Rational(1, scale)))
    r.always_positive = True
    return r


def f_bump(input: Variable, lb: Variable | int | float, ub: Variable | int | float, scale: int | float = 1000):
    """
    This is a convenience function that creates a bump function block.
    As it uses f_step it is also an approximation.
    lb and ub designate where the bump should (approximately) start and end.
    These the locations where the ReLU is located.
    So at lb the function value is 0 and rising to 1 with gradient being the scale.
    At ub the function value is 1 and decreasing to 0 with gradient being the scale.

    Works with scalar and vector-valued inputs.
    """

    # make sure that either all are variables or all are numbers
    if not all(isinstance(i, (int, float)) for i in [lb, ub]) and not all(isinstance(i, Variable) for i in [lb, ub]):
        raise ValueError(
            f"Expected both the upper and the lower bounds to be only either Variables or numbers, got {[i.__class__.__name__ for i in [lb, ub]]}"
        )

    # if all are variables make sure they have the same dimension
    if all(isinstance(i, Variable) for i in [lb, ub]) and not lb.dim == ub.dim == input.dim:
        raise ValueError(
            f"Expected both the upper and the lower bounds to have the same dimension as the input, got {[i.dim for i in [input, lb, ub]]}"
        )

    r = f_step(input - lb, scale=scale) - f_step(input - ub, scale=scale)
    r.always_positive = True
    return r


def f_eq(x: Variable | int | float, y: Variable | int | float, scale: int | float = 1000) -> Variable:
    """
    This is a convenience function that creates an equality function block.
    The precision is up to 1/scale because of us using ReLUs underneath.

    Works with scalar and vector-valued inputs. With vector-valued inputs, it acts element-wise.
    """

    # make sure that either all are variables or all are numbers
    if not all(isinstance(i, (int, float)) for i in [x, y]) and not all(isinstance(i, Variable) for i in [x, y]):
        raise ValueError(
            f"Expected all inputs to be only either Variables or numbers, got {[i.__class__.__name__ for i in [x, y]]}"
        )

    # if all are variables make sure they have the same dimension
    if all(isinstance(i, Variable) for i in [x, y]) and not y.dim == x.dim:
        raise ValueError(f"Expected both Variables to have the same dimension, got {[i.dim for i in [x, y]]}")

    r = f_step(-1 * (f_step(x - y, scale=scale) + f_step(y - x, scale=scale) - 1), scale=scale)
    r.always_positive = True
    return r


def f_neq(x: Variable | int | float, y: Variable | int | float, scale: int | float = 1000) -> Variable:
    """
    This is a convenience function that creates an unequality function block.
    The precision is up to 1/scale because of us using ReLUs underneath.

    Works with scalar and vector-valued inputs. With vector-valued inputs, it acts element-wise.

    """

    # make sure that either all are variables or all are numbers
    if not all(isinstance(i, (int, float)) for i in [x, y]) and not all(isinstance(i, Variable) for i in [x, y]):
        raise ValueError(
            f"Expected all inputs to be only either Variables or numbers, got {[i.__class__.__name__ for i in [x, y]]}"
        )

    # if all are variables make sure they have the same dimension
    if all(isinstance(i, Variable) for i in [x, y]) and not y.dim == x.dim:
        raise ValueError(f"Expected both Variables to have the same dimension, got {[i.dim for i in [x, y]]}")

    r = f_step(f_step(x - y, scale=scale) + f_step(y - x, scale=scale), scale=scale)
    r.always_positive = True
    return r


def f_not(x: Variable):
    """
    This is a convenience function that creates a NOT function block.
    It assumes that x is 0 or 1
    """

    r = 1 - x
    r.always_positive = True
    return r


def f_ifeq(
    x: Variable | int | float,
    y: Variable | int | float,
    t_val: Optional[Variable | int | float],
    f_val: Optional[Variable | int | float],
    scale: int | float = 1000,
    large_constant: int | float = 1e10,
) -> Variable:
    """
    This is a convenience function that creates an if-else block.
    If x==y, then t_val is returned, otherwise f_val is returned.

    The precision is up to 1/scale because of us using ReLUs underneath.
    Works with scalar and vector-valued inputs. With vector-valued inputs, it acts element-wise.
    """

    return f_ifelse(f_eq(x, y, scale=scale), t_val, f_val, large_constant=large_constant)


def f_ifelse(
    condition: Variable,
    t_val: Optional[Variable | int | float | sympy.Basic],
    f_val: Optional[Variable | int | float | sympy.Basic],
    large_constant: int | float = 1e10,
) -> Variable:
    """
    Conditional assignment operation implemented without Multi blocks.
    """

    assert t_val is not None or f_val is not None, "At least one of t_val or f_val should be provided"

    if t_val is None:
        return ReLU(condition * (-1 * large_constant) + f_val) - ReLU(condition * (-1 * large_constant) - f_val)

    if f_val is None:
        return +ReLU(f_not(condition) * (-1 * large_constant) + t_val) - ReLU(
            f_not(condition) * (-1 * large_constant) - t_val
        )

    return (
        ReLU(condition * (-1 * large_constant) + f_val)
        + ReLU(f_not(condition) * (-1 * large_constant) + t_val)
        - ReLU(condition * (-1 * large_constant) - f_val)
        - ReLU(f_not(condition) * (-1 * large_constant) - t_val)
    )


def f_ifelse_step(
    condition: Variable,
    t_val: Optional[Variable | int | float | sympy.Basic],
    f_val: Optional[Variable | int | float | sympy.Basic],
    large_constant: int | float = 1e10,
    scale: int | float = 1000,
) -> Variable:

    # fmt: off
    if t_val is None:
        return ReLU(-large_constant + large_constant * f_step(0.5 - condition, scale=scale) + f_val) \
             - ReLU(-large_constant + large_constant * f_step(0.5 - condition, scale=scale) - f_val)

    if f_val is None:
        
        return ReLU(-large_constant + large_constant * f_step(condition - 0.5, scale=scale) + t_val) \
             - ReLU(-large_constant + large_constant * f_step(condition - 0.5, scale=scale) - t_val)

    return ReLU(-large_constant + large_constant * f_step(0.5 - condition, scale=scale) + f_val) \
         - ReLU(-large_constant + large_constant * f_step(0.5 - condition, scale=scale) - f_val) \
         + ReLU(-large_constant + large_constant * f_step(condition - 0.5, scale=scale) + t_val) \
         - ReLU(-large_constant + large_constant * f_step(condition - 0.5, scale=scale) - t_val)
    # fmt: on


def f_ifelse_mul(
    condition: Variable,
    t_val: Optional[Variable | int | float | sympy.Basic],
    f_val: Optional[Variable | int | float | sympy.Basic],
) -> Variable:
    """
    Conditional using a multiplicative gate.
    Should be more stable than the one with only linear functions and ReLU activations.
    It assumes that all inputs have the same dimension!
    If t_val or f_val is not provided, it will be assumed to be 0
    """

    assert t_val is not None or f_val is not None, "At least one of t_val or f_val should be provided"

    # if t_val is zero, we can go without the linear and with a smaller concat
    if t_val is None:
        c = Concat([f_not(condition), f_val])
        m = Multiplicative(c)
        return m
    # same if f_val is zero
    if f_val is None:
        c = Concat([condition, t_val])
        m = Multiplicative(c)
        return m

    c = Concat([condition, f_not(condition), t_val, f_val])
    m = Multiplicative(c)
    l = Linear(input=m, A=Matrix.hstack(Matrix.eye(m.dim // 2), Matrix.eye(m.dim // 2)), b=Matrix.zeros(m.dim // 2, 1))
    return l


def f_larger(x: Variable | int | float, y: Variable | int | float, scale=1000) -> Variable:
    """
    This is a convenience function that creates a larger function block.
    If x > y, then 1 is returned, otherwise 0.

    The precision is up to 1/scale because of us using ReLUs underneath.

    Works with scalar and vector-valued inputs. With vector-valued inputs, it acts element-wise.
    """

    r = f_step(x - y, scale=scale)
    r.always_positive = True
    return r


def f_smaller(x: Variable | int | float, y: Variable | int | float, scale=1000) -> Variable:
    """
    This is a convenience function that creates a smaller function block.
    If x < y, then 1 is returned, otherwise 0.

    The precision is up to 1/scale because of us using ReLUs underneath.

    Works with scalar and vector-valued inputs. With vector-valued inputs, it acts element-wise.
    """

    r = f_step(y - x, scale=scale)
    r.always_positive = True
    return r


def f_and(x: Variable, y: Variable, scale=1000) -> Variable:
    """
    This is a convenience function that creates an AND function block.
    If x and y are both 1, then 1 is returned, otherwise 0.

    The precision is up to 1/scale because of us using ReLUs underneath.

    Works with scalar and vector-valued inputs. With vector-valued inputs, it acts element-wise.
    """

    r = ReLU(f_step(x, scale=scale) + f_step(y, scale=scale) - 1)
    r.always_positive = True
    return r


def f_or(x: Variable, y: Variable, scale=1000) -> Variable:
    """
    This is a convenience function that creates an OR function block.
    If x or y are 1, then 1 is returned, otherwise 0.
    """
    r = f_step(x + y, scale=scale)
    r.always_positive = True
    return r


def f_modulo_counter(dummy_input: Variable, divisor: int) -> Variable:
    """
    Computes the x mod divisor where x is a counter starting from zero.
    The idea is that we rotate a unit vector
    so that it makes a full revolution
    every divisor rotations.

    dummy_input can be any variable, we use it only to construct a constant.
    """

    angle = sympy.Integer(2) * sympy.pi / sympy.Integer(divisor)
    R = sympy.Matrix([[sympy.cos(angle), -sympy.sin(angle)], [sympy.sin(angle), sympy.cos(angle)]])

    unit_vector = sympy.Matrix([[1], [0]])

    # we first rotate, then output so if we want the first output to be 0,
    # we need to have the init_state one step before that
    init_state = Matrix(R.inv() @ unit_vector)

    # this rotates a 2D vector 1/divisor revolutions at a time
    cycler = LinState(
        input=dummy_input,
        A=Matrix(R),
        B=Matrix.zeros(2, dummy_input.dim),
        init_state=init_state,
    )

    # we now need to extract the position of the cycler

    extractor_matrix = Matrix.vstack(*[Matrix((R**i * unit_vector).T) for i in range(divisor)])
    indicator = Linear(input=cycler, A=extractor_matrix, b=Matrix.zeros(divisor, 1))

    # the dot product with the row of extractor_matrix corresponding to the current position of the cycler would be 1
    # the dot product with the second highest is sympy.cos(angle).
    # therefore, we can threshold at 1-sympy.cos(angle/2) to get a one hot encoding of the current position of the cycler

    one_hot = f_larger(indicator, sympy.cos(angle / sympy.Integer(2)))  # dim: divisor

    # and to get an integer value we need one final linear layer
    mod_value = Linear(one_hot, A=Matrix(sympy.Matrix(list(i for i in range(divisor))).T), b=Matrix.zeros(1, 1))
    mod_value.always_positive = True

    return mod_value
