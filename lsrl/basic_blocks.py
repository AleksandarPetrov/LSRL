from __future__ import annotations

import copy
import pickle
from typing import Any, List, Optional
from abc import ABC, abstractmethod

import os
import warnings
import numpy as np
import networkx as nx
import sympy

from lsrl.matrix_tools import Matrix, MatrixMode

from .utils import is_directed_path_graph


class GraphSimplificationError(Exception):
    pass


class Variable(ABC):

    def __init__(self, dim: int, name="Variable"):
        self.dim = dim
        self.input = None
        self.name = name
        self.hooks = {
            "before_forward": [],
            "after_forward": [],
        }

        self._loops_register = []
        self._graph = None
        self._topological_sort = None
        self.always_positive = False

    def graph(self, recompute=False):
        """Builds the compute graph for this variable"""
        if self._graph is None or recompute:
            self._graph = nx.DiGraph()
            stack = [self]
            already_processed = []
            while stack:
                node = stack.pop()
                self._graph.add_node(node)
                already_processed.append(node)

                if node.input is not None:
                    if isinstance(node.input, list) and all(isinstance(x, Variable) for x in node.input):
                        for i in node.input:
                            self._graph.add_edge(i, node)
                            if i not in already_processed:
                                stack.append(i)

                    elif isinstance(node.input, Variable):
                        self._graph.add_edge(node.input, node)
                        stack.append(node.input)
                    else:
                        raise TypeError(f"Invalid input type {type(node.input)}")

            self._topological_sort = list(nx.topological_sort(self._graph))
        return self._graph

    def topological_sort(self, recompute=False):
        """Gives the topological sort of the compute graph."""
        if self._topological_sort is None or recompute:
            self.graph(recompute=True)

        return self._topological_sort

    def __call__(self, *args, **kwargs):

        for h in self.hooks["before_forward"]:
            h(self, *args, **kwargs)

        out = self.forward(*args, **kwargs)

        for h in self.hooks["after_forward"]:
            h(self, out, *args, **kwargs)

        return out

    @abstractmethod
    def forward(self) -> Matrix:
        raise NotImplementedError("Forward method not implemented")

    def __add__(self, other: Variable | int | float | sympy.Basic) -> Variable:

        if isinstance(other, Variable):
            assert self.dim == other.dim, f"Dimension mismatch for sum: {self.dim} != {other.dim}"
            both = Concat([self, other])
            r = Linear(
                input=both,
                A=Matrix.hstack(Matrix.eye(self.dim), Matrix.eye(self.dim)),
                b=Matrix.zeros(self.dim, 1),
                name="Sum",
            )
            if self.always_positive and other.always_positive:
                r.always_positive = True

        elif isinstance(other, (int, float, sympy.Basic)):
            if not isinstance(other, sympy.Basic):
                other = sympy.Rational(other)

            r = Linear(
                input=self,
                A=Matrix.eye(self.dim),
                b=Matrix.ones(self.dim, 1) * other,
                name="Sum",
            )
            if self.always_positive and other >= 0:
                r.always_positive = True
        else:
            raise ValueError(f"Unsupported type for addition: {type(other)}")

        return r

    def __radd__(self, other: Variable | int | float | sympy.Basic) -> Variable:
        return self.__add__(other)

    def __sub__(self, other: Variable | int | float | sympy.Basic) -> Variable:

        if isinstance(other, Variable):
            assert self.dim == other.dim, f"Dimension mismatch for sum: {self.dim} != {other.dim}"
            both = Concat([self, other])
            return Linear(
                input=both,
                A=Matrix.hstack(Matrix.eye(self.dim), (-sympy.core.numbers.One()) * Matrix.eye(self.dim)),
                b=Matrix.zeros(self.dim, 1),
                name="Sub",
            )
        elif isinstance(other, (int, float, sympy.Basic)):
            if not isinstance(other, sympy.Basic):
                other = sympy.Rational(other)
            return Linear(
                input=self,
                A=Matrix.eye(self.dim),
                b=Matrix.ones(self.dim, 1) * -other,
                name="Sub",
            )
        else:
            raise ValueError(f"Unsupported type for addition: {type(other)}")

    def __rsub__(self, other: Variable | int | float | sympy.Number) -> Variable:
        if isinstance(other, Variable):
            assert self.dim == other.dim, f"Dimension mismatch for sum: {self.dim} != {other.dim}"
            both = Concat([self, other])
            return Linear(
                input=both,
                A=Matrix.hstack((-sympy.core.numbers.One()) * Matrix.eye(self.dim), Matrix.eye(self.dim)),
                b=Matrix.zeros(self.dim, 1),
                name="Sub",
            )
        elif isinstance(other, (int, float, sympy.Basic)):
            if not isinstance(other, sympy.Basic):
                other = sympy.Rational(other)
            return Linear(
                input=self,
                A=(-sympy.core.numbers.One()) * Matrix.eye(self.dim),
                b=Matrix.ones(self.dim, 1) * other,
                name="Sub",
            )
        else:
            raise ValueError(f"Unsupported type for addition: {type(other)}")

    def __mul__(self, other: int | float | sympy.Basic) -> Variable:
        if not isinstance(other, sympy.Basic):
            other = sympy.Rational(other)
        if isinstance(other, (int, float, sympy.Basic)):
            r = Linear(
                input=self,
                A=Matrix.eye(self.dim) * other,
                b=Matrix.zeros(self.dim, 1),
                name="Scale",
            )
            if self.always_positive and other >= 0:
                r.always_positive = True
            return r
        else:
            raise TypeError(f"Unsupported scalar for multiplication: {type(other)}")

    def __rmul__(self, other: int | float | sympy.Number) -> Variable:
        return self.__mul__(other)

    def __truediv__(self, other: int | float | sympy.Basic) -> Variable:
        if isinstance(other, (int, float, sympy.Basic)):
            if other == 0:
                raise ValueError("Division by zero!")
            other = sympy.Number(other)
            return self.__mul__(sympy.Integer(1) / other)
        else:
            raise TypeError(f"Unsupported scalar for division: {type(other)}")

    def __getitem__(self, key: int | slice) -> Variable:
        if isinstance(key, int):
            return Slice(self, key, key + 1)
        elif isinstance(key, slice):
            # we do not support steps
            if key.step is not None:
                raise ValueError("Slice does not support steps")

            return Slice(
                self,
                key.start if key.start is not None else 0,
                key.stop if key.stop is not None else self.dim,
            )
        else:
            raise ValueError(f"Unsupported type for slicing: {type(key)}")

    def prepare_variable(self, x: np.array | list | Matrix | sympy.Basic) -> Matrix:
        """
        Ensure that a variable is in the right type/format.
        If numeric it should be a numpy array, if not, should be a sympy Matrix.
        Regardless of the type, it should be a column vector represented as a Matrix
        """

        if isinstance(x, sympy.MatrixBase):
            x = Matrix(x)

        if isinstance(x, (float, int)):
            x = np.array([[x]])

        if isinstance(x, (np.ndarray, list)):
            if len(x.shape) == 1:
                x = x[:, None]
            x = Matrix(x)

        if isinstance(x, Matrix):
            if x.cols == 1:
                return x
            else:
                raise ValueError(f"Expected a column vector but got shape {x.shape}")

        raise ValueError(f"Invalid input type {type(x)}")


class Input(Variable):
    def __init__(self, dim: int, name="Input"):
        super().__init__(dim, name=name)
        self.size_str = str(self.dim)

    def forward(self, x: Matrix) -> Matrix:
        return x


class ForEach:
    def __init__(self, output: Variable, keep_graph_history=False, verbose=True):
        self.output = output
        self.keep_graph_history = keep_graph_history
        self.verbose = verbose
        self._initialize()

    def _initialize(self):

        # the input is the first node in a topological sort as it has 0 in-degree
        self.input = self.topological_sort()[0]
        self.simplification_history: List[nx.DiGraph] = []

    def __getstate__(self):
        """Used for serializing to a file"""
        return {
            "output": self.output,
            "keep_graph_history": self.keep_graph_history,
            "verbose": self.verbose,
        }

    def __setstate__(self, attributes):
        """Used for deserializing from a file"""
        self.__dict__.update(attributes)
        self._initialize()

    def record_simplification_step(self, name: str = "unknown") -> None:
        """
        Records the simplifcation step into the history and
        also triggers the update of the computation graph.
        """
        self.graph(recompute=True)
        new_graph = self.graph().copy() if self.keep_graph_history else None
        self.simplification_history.append((f"{name}", new_graph))

        if self.verbose:
            node_types = [n.__class__.__name__ for n in self.graph().nodes]
            max_outdegree = max(self.graph().out_degree(n) for n in self.graph().nodes)
            # get counts for each differnet type
            counts = {n: node_types.count(n) for n in set(node_types)}
            # format as a string
            counts_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])

            try:
                branch_node = next(n for n in self.topological_sort() if self.graph().out_degree(n) > 1)
                progress = f"{nx.shortest_path_length(self.graph(), self.input, branch_node)}/{nx.shortest_path_length(self.graph(), self.input, self.output)}"
            except StopIteration:
                progress = f"{nx.shortest_path_length(self.graph(), self.input, self.output)}/{nx.shortest_path_length(self.graph(), self.input, self.output)}"

            print(
                f"{len(self.simplification_history):>5}. {name} {counts_str} (Total: {self.graph().number_of_nodes()} nodes). MaxBranches: {max_outdegree}. Progress: {progress}"
            )

    def graph(self, recompute=False):
        """Builds the compute graph for the output variable"""
        return self.output.graph(recompute)

    def topological_sort(self, recompute=False):
        """Gives the topological sort of the compute graph."""
        return self.output.topological_sort(recompute)

    def get_state_variables(self):
        """Traverses the graph and get all LinState variables."""
        linear_states = []
        for node in self.topological_sort():
            if isinstance(node, LinState):
                linear_states.append(node)
        return linear_states

    def reset(self):
        """Sets all state variables to their initial value. Useful when evaluating on multiple sequences."""
        for state_var in self.get_state_variables():
            state_var.reset()

    def _call_single_input(self, x: Matrix):

        assert x.shape == (
            (self.input.dim, 1)
        ), f"Input size mismatch: ForEach layer expected {(self.input.dim)} but got {x.shape}"

        results = {}

        # A topological sort gives us nodes in the order they should be processed
        for node in self.topological_sort():
            func = node.__call__

            # If no predecessors, then we have the input node and its value is computed directly:
            if len(list(self.graph().predecessors(node))) == 0:
                results[node] = func(x)

            elif isinstance(node.input, Variable):
                results[node] = func(results[node.input])
            elif isinstance(node.input, list) and all(isinstance(x, Variable) for x in node.input):
                results[node] = func(*[results[x] for x in node.input])
            else:
                raise ValueError(f"Invalid input type {type(node.input)}")

        return results[self.output]

    def _call_multiple_inputs(self, x: Matrix):

        self.reset()

        outputs = []
        for col_idx in range(x.cols):
            output = self._call_single_input(x[:, col_idx])
            outputs.append(output)

        return Matrix.hstack(*outputs)

    def __call__(self, x: Matrix, **kwargs):
        """
        x should be either a column vector with length corresponding to the input dimension or
        an array of inputs with each column corresponding to one input.
        The output will be similarly formatted: individual elements correspond to columns.
        Note that this is different from the usual case where rows correspond to individual elements.
        This is done in order to match the matrix multiplication convention.
        """

        if isinstance(x, (float, int)):
            x = Matrix([[x]])
        elif isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                x = Matrix(x[:, None])
            elif len(x.shape) == 2:
                x = Matrix(x)
            else:
                raise ValueError(f"Invalid input ndarray shape {x.shape}")

        if not isinstance(x, Matrix):
            raise ValueError(f"Invalid input type {type(x)}")

        if x.rows != self.input.dim:
            raise ValueError(f"Input size mismatch: expected input dimension {self.input.dim} but got {x.rows}")

        if x.cols == 1:
            return self._call_single_input(x, **kwargs)
        elif x.cols > 1:
            return self._call_multiple_inputs(x, **kwargs)
        else:
            raise ValueError(f"Invalid input shape {x.shape}")

    def validate_graph(self):
        # make sure only one input variable in the graph
        input_nodes = [n for n in self.graph().nodes if isinstance(n, Input)]
        if len(input_nodes) != 1:
            raise ValueError(f"Invalid number of input nodes: {len(input_nodes)}! Must be exactly 1.")

        # check that the graph is a DAG
        if not nx.is_directed_acyclic_graph(self.graph()):
            raise ValueError("The computation graph is not a directed acyclic graph!")

        # TODO: check that the input and output sizes match

    @property
    def parameter_count(self):
        """Number of `learnable` parameters."""
        count = 0
        # iterate over the nodes:
        for node in self.topological_sort():
            if isinstance(node, (Linear, Slice)):
                count += node.A.rows * node.A.cols + node.b.rows * node.b.cols
            elif isinstance(node, LinState):
                count += node.A.rows * node.A.cols + node.B.rows * node.B.cols + node.bias.rows * node.bias.cols
        return count

    # TODO: Make this into a nicer serialization method
    # that only keeps the data but not the class methods
    def save(self, filename):
        # create the directory if it doesn't exist
        if "/" in filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(self, filename):
        with open(filename, "rb") as f:
            loaded = pickle.load(f)
        return loaded

    def debranch_graph(self, max_steps=np.inf):
        """Recursively remove all the branches of the computation graph by following the debranching rules."""

        self.record_simplification_step("before_debranching")

        # check if the graph actually has branches
        while not is_directed_path_graph(self.graph()) and len(self.simplification_history) < max_steps:

            # find the first branching point
            for branch_node in self.topological_sort():
                if len(list(self.graph().successors(branch_node))) > 1:
                    break

            # get all the variables directly after the branch
            succs = list(self.graph().successors(branch_node))

            # Case 1A: If all are Multiplicative then they are computing the exact same thing and only one can be kept
            if all(isinstance(s, Multiplicative) for s in succs):
                new_mul = Multiplicative(branch_node)
                for s in succs:
                    self._replace_in_successors(s, new_mul)
                self.record_simplification_step("only_multiplicatives")

            # Case 1B: If any Multiplicative nodes are present, deal with them first by bypassing the other nodes
            elif any(isinstance(s, Multiplicative) for s in succs):
                hs = branch_node.dim // 2  # the multiplicative output size
                # fmt: off
                bypass = Multiplicative(
                    Linear(
                        input = branch_node,
                        A=Matrix.vstack( 
                            Matrix.hstack(Matrix.eye(hs), Matrix.zeros(hs, hs)),
                            Matrix.hstack(Matrix.zeros(2*hs, hs), Matrix.zeros(2*hs, hs)),
                            Matrix.hstack(Matrix.zeros(hs, hs), Matrix.eye(hs)),
                            Matrix.hstack(Matrix.eye(hs), Matrix.zeros(hs, hs)),
                            Matrix.hstack(Matrix.zeros(hs, hs), Matrix.eye(hs)),
                        ),
                        b=Matrix.vstack(
                            Matrix.zeros(hs, 1),
                            Matrix.ones(2*hs, 1),
                            Matrix.zeros(3*hs, 1),
                        ),
                        name="BypassOfMul",
                    )
                )
                # fmt: on
                new_mul = Slice(bypass, 0, hs)
                orig_input = Slice(bypass, hs, 3 * hs)

                for s in succs:
                    if isinstance(s, Multiplicative):
                        self._replace_in_successors(s, new_mul)
                    else:
                        self._replace_in_successors(s, orig_input)
                self.record_simplification_step("multiplicative_bypass")

            # Case 2: If they are all State variables we can combine them into a single State
            elif all(isinstance(s, LinState) for s in succs):
                combined_linstate = LinState(
                    input=branch_node,
                    A=Matrix.diag(*[s.A for s in succs]),
                    B=Matrix.vstack(*[s.B for s in succs]),
                    init_state=Matrix.vstack(*[s.init_state for s in succs]),
                    bias=Matrix.vstack(*[s.bias for s in succs]),
                    name="+".join([s.name for s in succs]),
                )

                # add a slice of the state for the variables that depend on each of the individual LinStates:
                for s in succs:
                    slice = Slice(
                        input=combined_linstate,
                        start=sum([x.dim for x in succs[: succs.index(s)]]),
                        end=sum([x.dim for x in succs[: succs.index(s) + 1]]),
                        name=f"Slice_{s.name}",
                    )
                    self._replace_in_successors(s, slice)

                self.record_simplification_step("only_states")
                self.fold_same_type()

            # Case 3: If they are all ReLUs we can combine them into a single ReLU
            elif all(isinstance(s, ReLU) for s in succs):
                new_relu = ReLU(branch_node)
                for s in succs:
                    self._replace_in_successors(s, new_relu)
                self.record_simplification_step("only_relus")
                self.fold_same_type()

            # Case 4: If they are all Concats we can convert them into a Linear as they must all be just copying the input
            elif all(isinstance(s, Concat) for s in succs):

                # One complication is that concatts can depend on other concats, for example:
                #
                # BranchNode --------------┑
                #    |   ┖---------┑     Concat---
                #    |           Concat----┙
                #    ┖-------------┙
                #
                # So, we will restrict ourselves at this step by only treating the concats that depend only on the branch node directly
                # The rest will be handled by the Linear and Concat Case or the Linear and linear case

                concats_depending_only_on_branch_node = [
                    s for s in succs if all(concat_input == branch_node for concat_input in s.input)
                ]
                for concat in concats_depending_only_on_branch_node:
                    n_copies = len(concat.input)
                    new_linear = Linear(
                        input=branch_node,
                        A=Matrix.vstack(*[Matrix.eye(branch_node.dim) for _ in range(n_copies)]),
                        b=Matrix.zeros(branch_node.dim * n_copies, 1),
                        name=f"Duplicate",
                    )
                    self._replace_in_successors(concat, new_linear)
                self.record_simplification_step("concat_to_duplicate")

            # Case 5A: Only Linear and they are all slices
            # We will push the slices into the next blocks
            elif all(isinstance(s, Slice) for s in succs):

                # Slice and Multiply... this is difficult cus we can't push the slice after the multiply.
                # So, we will instead create a Multiply bypass.
                # then we will need to rerun this funcion
                multiplies_after_slices = [
                    ss
                    for s in list(self.graph().successors(branch_node))
                    for ss in list(self.graph().successors(s))
                    if isinstance(s, Slice) and isinstance(ss, Multiplicative)
                ]
                if len(multiplies_after_slices) > 0:
                    slices_of_multiplies_after_slices = [m.input for m in multiplies_after_slices]
                    assert all(isinstance(s, Slice) for s in slices_of_multiplies_after_slices)

                    slices_of_non_multiplies_after_slices = [
                        s
                        for s in list(self.graph().successors(branch_node))
                        for ss in list(self.graph().successors(s))
                        if isinstance(s, Slice) and not isinstance(ss, Multiplicative)
                    ]

                    pre_mul_linear = Linear(
                        input=branch_node,
                        A=Matrix.vstack(
                            *(
                                [s.A[: s.A.rows // 2, :] for s in slices_of_multiplies_after_slices]
                                + [Matrix.zeros(branch_node.dim, branch_node.dim)]
                                + [s.A[s.A.rows // 2 :, :] for s in slices_of_multiplies_after_slices]
                                + [Matrix.eye(branch_node.dim)]
                            )
                        ),
                        b=Matrix.vstack(
                            *(
                                [Matrix.zeros(s.A.rows // 2, 1) for s in slices_of_multiplies_after_slices]
                                + [Matrix.ones(branch_node.dim, 1)]
                                + [Matrix.zeros(s.A.rows // 2, 1) for s in slices_of_multiplies_after_slices]
                                + [Matrix.zeros(branch_node.dim, 1)]
                            )
                        ),
                        name="PreMulLinear",
                    )

                    mul = Multiplicative(pre_mul_linear)
                    mul_slices = [
                        Slice(
                            input=mul,
                            start=sum([s.dim for s in slices_of_multiplies_after_slices[:i]]) // 2,
                            end=sum([s.dim for s in slices_of_multiplies_after_slices[: i + 1]]) // 2,
                        )
                        for i in range(len(multiplies_after_slices))
                    ]

                    passthrough = Slice(
                        input=mul,
                        start=sum([s.dim for s in slices_of_multiplies_after_slices]) // 2,
                        end=mul.dim,
                    )

                    # replace the slices in multiplies_after_slices with the corresponding in mul_slices
                    for s, new_s in zip(multiplies_after_slices, mul_slices):
                        self._replace_in_successors(s, new_s)
                    # replace the input of the other slices to be the passthrough
                    for s in slices_of_non_multiplies_after_slices:
                        s.input = passthrough

                    self.record_simplification_step("slice_into_next_multiplicative")
                    continue

                # push slices into the following Linear blocks
                for linear_node in [
                    ss
                    for s in list(self.graph().successors(branch_node))
                    for ss in list(self.graph().successors(s))
                    if isinstance(s, Slice) and isinstance(ss, Linear)
                ]:
                    slice = list(self.graph().predecessors(linear_node))[0]
                    assert (
                        slice.input == branch_node
                    ), f"Slice must follow the branch node. Slice input is {slice.input} while branch node is {branch_node}"
                    # combine the two linear layers
                    new_linear = Linear(
                        input=branch_node,
                        A=linear_node.A @ slice.A,
                        b=linear_node.A @ slice.b + linear_node.b,
                        name=f"{slice.name}+{linear_node.name}",
                    )

                    self._replace_in_successors(linear_node, new_linear)

                # push slices into the following State blocks
                for state_node in [
                    ss
                    for s in list(self.graph().successors(branch_node))
                    for ss in list(self.graph().successors(s))
                    if isinstance(s, Slice) and isinstance(ss, LinState)
                ]:
                    slice = list(self.graph().predecessors(state_node))[0]
                    assert (
                        slice.input == branch_node
                    ), f"Slice must follow the branch node. Slice input is {slice.input} while branch node is {branch_node}"
                    # combine the two linear layers
                    new_state = LinState(
                        input=branch_node,
                        A=state_node.A,
                        B=state_node.B @ slice.A,
                        init_state=state_node.init_state,
                        bias=state_node.B @ slice.b + state_node.bias,
                        name=f"{slice.name}+{state_node.name}",
                    )

                    self._replace_in_successors(state_node, new_state)

                # ReLU and slice can switch places
                for relu_node in [
                    ss
                    for s in list(self.graph().successors(branch_node))
                    for ss in list(self.graph().successors(s))
                    if isinstance(s, Slice) and isinstance(ss, ReLU)
                ]:
                    slice = list(self.graph().predecessors(relu_node))[0]
                    assert (
                        slice.input == branch_node
                    ), f"Slice must follow the branch node. Slice input is {slice.input} while branch node is {branch_node}"

                    new_relu = ReLU(branch_node)
                    new_slice = Slice(new_relu, start=slice.start, end=slice.end)
                    relu_node.input = branch_node
                    self._replace_in_successors(relu_node, new_slice)

                # push slice after Concat as a linear layer
                for concat_node in set(
                    [
                        ss
                        for s in list(self.graph().successors(branch_node))
                        for ss in list(self.graph().successors(s))
                        if isinstance(s, Slice) and isinstance(ss, Concat)
                    ]
                ):  # the same concat might have multiple copies of the branching point in its input

                    inputs = [(x, isinstance(x, Slice) and x.input == branch_node) for x in concat_node.input]
                    input_concat = Concat(
                        [x if not is_slice_of_bn else branch_node for x, is_slice_of_bn in inputs]
                    )  # TODO: this is not efficient as we end up having potentially multiple copies of branch_node
                    post_concat_linear = Linear(
                        input=input_concat,
                        # A=sympy.SparseMatrix.diag(
                        #     *[x.A if is_slice_of_bn else sympy.SparseMatrix.eye(x.dim) for x, is_slice_of_bn in inputs]
                        # ),
                        A=Matrix.diag(*[x.A if is_slice_of_bn else x.dim for x, is_slice_of_bn in inputs]),
                        b=Matrix.vstack(
                            *[x.b if is_slice_of_bn else Matrix.zeros(x.dim, 1) for x, is_slice_of_bn in inputs]
                        ),
                        name="SliceAftConcat",
                    )

                    self._replace_in_successors(concat_node, post_concat_linear)

                self.record_simplification_step("slice_into_next")
                self.fold_same_type()

            # Case 5B: Only Linear and they are not all slices
            # We can combine them into a single Linear layer and then add
            # slices which can be pushed into the next operations using Case 5A
            elif all(isinstance(s, Linear) for s in succs):

                combined_linear = Linear(
                    input=branch_node,
                    A=Matrix.vstack(*[s.A for s in succs]),
                    b=Matrix.vstack(*[s.b for s in succs]),
                    name="+".join([s.name for s in succs]),
                )

                # add a slice for each linear layer
                for s in succs:
                    slice = Slice(
                        input=combined_linear,
                        start=sum([x.dim for x in succs[: succs.index(s)]]),
                        end=sum([x.dim for x in succs[: succs.index(s) + 1]]),
                        name=f"Slice_{s.name}",
                    )
                    self._replace_in_successors(s, slice)

                self.record_simplification_step("only_linear_nonslice")
                self.fold_same_type()

            # Case 6: If both State variables and other variables are present, we can pass through the other variables with dummy State variables
            # Then, we can call this function again in order to fuse the State variables
            elif any(isinstance(s, LinState) for s in succs):
                passthrough = LinState(
                    input=branch_node,
                    A=Matrix.zeros(branch_node.dim, branch_node.dim),
                    B=Matrix.eye(branch_node.dim),
                    init_state=Matrix.zeros(branch_node.dim, 1),
                    bias=Matrix.zeros(branch_node.dim, 1),
                    name=f"Pass[{branch_node.name}]",
                )
                for var in [s for s in succs if not isinstance(s, LinState)]:
                    # create a passthrough state for the input of this var
                    if isinstance(var.input, Variable):
                        var.input = passthrough
                    elif isinstance(var.input, list):
                        var.input = [passthrough if x == branch_node else x for x in var.input]
                    else:
                        raise GraphSimplificationError(f"Invalid successor type {type(var.input)}")
                self.record_simplification_step("passthrough_nonstate_vars")
                self.fold_same_type()

            # Case 7A: Only Linear and ReLU where all Linear are followed by only one node that is a ReLU
            # If we add Liner bypasses to the ReLUs we will have all linear now, and all ReLU after that which we can remove with the above cases
            elif all(
                isinstance(s, ReLU)
                or (
                    isinstance(s, Linear)
                    and self.graph().out_degree(s) == 1
                    and isinstance(list(self.graph().successors(s))[0], ReLU)
                )
                for s in succs
            ):
                relus = [s for s in succs if isinstance(s, ReLU)]
                passthrough = Linear(
                    input=branch_node,
                    A=Matrix.eye(branch_node.dim),
                    b=Matrix.zeros(branch_node.dim, 1),
                    name=f"Pass[{branch_node.name}]",
                )
                for r in relus:
                    r.input = passthrough
                self.record_simplification_step("linear_relu_only__bypass_relus")

            # Case 7B: Only Linear and ReLU where some Linear are not followed by only one node that is a ReLU
            # Add ReLU bypasses for the linear layers then. We will be in 13A first, then 4B, then 2
            elif all(isinstance(s, (ReLU, Linear)) for s in succs):
                linear = [
                    s
                    for s in succs
                    if isinstance(s, Linear)
                    and not (self.graph().out_degree(s) == 1 and isinstance(list(self.graph().successors(s))[0], ReLU))
                ]
                passthrough = f_relu_identity(branch_node)
                for l in linear:
                    l.input = passthrough
                self.record_simplification_step("linear_relu_only__bypass_linear")

            # 8: Only Linear and Concat layers
            # Add linear bypasses for the concat layers which can then be merged in a subsequent pass with 4B and then 4A
            elif all(isinstance(s, (Linear, Concat)) for s in succs):

                passthrough = Linear(
                    input=branch_node,
                    A=Matrix.eye(branch_node.dim),
                    b=Matrix.zeros(branch_node.dim, 1),
                    name=f"Pass[{branch_node.name}]",
                )

                concats = set(
                    [s for s in succs if isinstance(s, Concat)]
                )  # the same concat might have multiple copies of the branching point in its input
                for c in concats:
                    # replace that input of the concat with the linear layer
                    c.input = [passthrough if x == branch_node else x for x in c.input]
                self.record_simplification_step("linear_and_concat_only")

            # 9: Only ReLU and Concat layers
            # Same strategy but with ReLU bypasses
            elif all(isinstance(s, (ReLU, Concat)) for s in succs):

                passthrough = f_relu_identity(branch_node)

                concats = set(
                    [s for s in succs if isinstance(s, Concat)]
                )  # the same concat might have multiple copies of the branching point in its input
                for c in concats:
                    # replace that input of the concat with the linear layer
                    c.input = [passthrough if x == branch_node else x for x in c.input]
                self.record_simplification_step("relu_and_concat_only")

            # 10: Linear, ReLU and Concat layers
            # We introduce ReLU bypasses to all Concat branches and to the Linear branches which are not immediately followed by a ReLU
            elif all(isinstance(s, (Linear, ReLU, Concat)) for s in succs):
                passthrough = f_relu_identity(branch_node)

                for c in set([s for s in succs if isinstance(s, Concat)]):
                    c.input = [passthrough if x == branch_node else x for x in c.input]
                self.record_simplification_step("linear_relu_and_concat__pass_concat")

                for l in [
                    s
                    for s in succs
                    if isinstance(s, Linear)
                    and self.graph().out_degree(s) == 1
                    and not isinstance(list(self.graph().successors(s))[0], ReLU)
                ]:
                    l.input = passthrough

                self.record_simplification_step("linear_relu_and_concat__pass_linear")

            else:
                raise GraphSimplificationError(
                    f"Invalid combination of branching successors: {[s.__class__.__name__ for s in succs]}"
                )

    def fold_same_type(self):
        """
        Perform some simple simplifications:
        1. Fuse consecutive Linear layers into a single one.
        2. Fuse consecutive Linear layers into a single one.
        3. Fuse Linear layers into the subsequent LinState layer.
        4. Replace Concat layers with Linear layers when all the inputs of the Concat layer are the same.
        """
        changed = True
        while changed:
            changed = False
            # for node in self.topological_sort():
            for node in nx.dfs_preorder_nodes(self.graph(), source=self.input):
                preds = list(self.graph().predecessors(node))
                if len(preds) == 1:
                    prev = preds[0]
                    if isinstance(node, Linear) and isinstance(prev, Linear):
                        if len(list(self.graph().successors(prev))) == 1:
                            # combine the two linear layers
                            new_linear = Linear(
                                input=prev.input,
                                A=node.A @ prev.A,
                                b=node.A @ prev.b + node.b,
                                name=f"{prev.name}+{node.name}",
                            )

                            self._replace_in_successors(node, new_linear)
                            self.record_simplification_step("fold_conseq_linear")
                            changed = True
                            break

                    if isinstance(node, ReLU) and isinstance(prev, ReLU):
                        node.input = preds[0].input
                        self.record_simplification_step("fold_conseq_relus")
                        changed = True
                        break

                    if isinstance(node, LinState) and isinstance(prev, Linear):
                        new_state = LinState(
                            input=prev.input,
                            A=node.A,
                            B=node.B @ prev.A,
                            init_state=node.init_state,
                            bias=node.B @ prev.b + node.bias,
                            name=f"{prev.name}+{node.name}",
                        )

                        self._replace_in_successors(node, new_state)
                        self.record_simplification_step("fold_state_after_linear")
                        changed = True
                        break

                    if isinstance(node, Concat) and all(concat_input == prev for concat_input in node.input):
                        n_copies = len(node.input)
                        new_linear = Linear(
                            input=prev,
                            A=Matrix.vstack(*[Matrix.eye(prev.dim) for _ in range(n_copies)]),
                            b=Matrix.zeros(prev.dim * n_copies, 1),
                            name=f"Duplicate",
                        )
                        self._replace_in_successors(node, new_linear)
                        self.record_simplification_step("fold_concat_to_duplicate")
                        changed = True
                        break

    def simplify(self, max_steps=np.inf):
        """Simplify the graph by removing all the branches and fusing subsequent layers whenever possible."""
        self.fold_same_type()
        self.debranch_graph(max_steps=max_steps)
        self.fold_same_type()

    def _replace_in_successors(self, old, new):
        """Find all variables that depend on old and replace this dependency with new."""

        # if no successors, then that means that this is our final variable.
        # in this case we need to replace our self.output and revalidate the graph
        if len(list(self.graph().successors(old))) == 0:
            self.output = new
            self.graph(recompute=True)
            self.validate_graph()

        else:
            for succ in self.graph().successors(old):
                # if a variable with a single input, i.e., Linear, ReLU, Slice, LinState, Multi
                if isinstance(succ.input, Variable):
                    succ.input = new
                # if Concat
                elif isinstance(succ.input, list) and all(isinstance(x, Variable) for x in succ.input):
                    succ.input = [new if x == old else x for x in succ.input]
                else:
                    raise GraphSimplificationError(f"Invalid successor type {type(succ.input)}")


class Linear(Variable):
    def __init__(
        self,
        input: Variable,
        A: Matrix,
        b: Matrix,
        name="Linear",
    ):
        if not isinstance(A, Matrix):
            raise ValueError(f"Invalid type for A: {type(A)}")
        if not isinstance(b, Matrix):
            raise ValueError(f"Invalid type for b: {type(b)}")
        if A.mode != b.mode:
            raise ValueError(f"Matrix A and b must have the same mode: {A.mode} != {b.mode}")

        super().__init__(A.rows, name=name)

        self.input = input
        self.size_str = f"{A.cols}->{A.rows}"

        assert A.cols == self.input.dim, f"A needs to have dim of anything x {self.input.dim} but it has {A.shape}"
        assert b.shape == (
            self.dim,
            1,
        ), f"b needs to have dim of {(self.dim,1)} but it has {b.shape}"

        self.A, self.b = A, b

    def forward(self, x: sympy.MatrixBase | np.array) -> Matrix:

        x = self.prepare_variable(x)

        assert x.shape == (
            self.A.cols,
            1,
        ), f"Input size mismatch: Linear layer expected {self.A.cols,1} but got {x.shape}"

        return self.A @ x + self.b


class Slice(Linear):
    def __init__(self, input: Variable, start: int, end: int, name=None):

        if name is None:
            name = f"Slice[{start}:{end}]"

        assert start < end, f"Invalid slice indices: start={start} end={end}"
        assert end <= input.dim, f"Invalid slice indices: end={end} exceeds input dim {input.dim}"

        # represent it as a linear layer
        dim = end - start

        # A = sympy.SparseMatrix.zeros(dim, input.dim)
        # A[:, start:end] = sympy.SparseMatrix.eye(dim)

        # this is about 100 times faster than the above method:
        A = Matrix.SliceMatrix(input_dim=input.dim, start=start, end=end)
        b = Matrix.zeros(dim, 1)
        self.start = start
        self.end = end

        super().__init__(input, A=A, b=b, name=name)

        self.always_positive = input.always_positive


class Concat(Variable):
    def __init__(self, inputs: List[Variable]):
        super().__init__(sum([i.dim for i in inputs]), name=f"Concat")
        self.input = inputs
        self.size_str = f"({','.join(str(i.dim) for i in self.input)})->{self.dim}"
        if all(i.always_positive for i in inputs):
            self.always_positive = True

    def forward(self, *args) -> Matrix:
        args = [self.prepare_variable(a) for a in args]
        assert len(args) == len(
            self.input
        ), f"Input size mismatch: Concat layer expected {len(self.input)} but got {len(args)}"
        assert all(
            [x.shape == (i.dim, 1) for x, i in zip(args, self.input)]
        ), f"Input size mismatch: Concat layer expected {[i.dim for i in self.input]} but got {[x.shape for x in args]}"

        return Matrix.vstack(*args)


class ReLU(Variable):
    def __init__(self, input: Variable):
        super().__init__(input.dim, name=f"ReLU")
        self.input = input
        self.size_str = f"{self.dim}->{self.dim}"
        self.always_positive = True

    def forward(self, x: Matrix) -> Matrix:
        x = self.prepare_variable(x)

        assert x.shape == (
            self.dim,
            1,
        ), f"Input size mismatch: ReLU layer expected {(self.dim, 1)} but got {x.shape}"

        return x.ReLU()


class LinState(Variable):
    def __init__(
        self,
        input: Variable,
        A: sympy.Basic,
        B: sympy.Basic,
        init_state: sympy.Basic,
        name="LinState",
        bias=None,
    ):

        if not isinstance(A, Matrix):
            raise ValueError(f"Invalid type for A: {type(A)}")
        if not isinstance(B, Matrix):
            raise ValueError(f"Invalid type for b: {type(B)}")
        if not isinstance(init_state, Matrix):
            raise ValueError(f"Invalid type for init_state: {type(init_state)}")

        super().__init__(dim=A.rows, name=name)

        if bias is None:
            bias = Matrix(sympy.SparseMatrix.zeros(self.dim, 1), skip_symbolic_check=True)

        if not isinstance(bias, Matrix):
            raise ValueError(f"Invalid type for bias: {type(bias)}")

        if A.mode != B.mode or A.mode != init_state.mode or (bias is not None and A.mode != bias.mode):
            raise ValueError(f"Matrices A, B, init_state and bias must have the same mode")

        self.input = input
        self.size_str = f"{B.cols}->{self.dim}"

        assert bias.shape == (
            self.dim,
            1,
        ), f"bias needs to have shape of {self.dim} x 1 but it has {bias.shape}"
        assert A.shape == (
            self.dim,
            self.dim,
        ), f"A needs to have shape of {self.dim} x {self.dim} but it has {A.shape}"
        assert B.shape == (
            self.dim,
            self.input.dim,
        ), f"B needs to have shape of {self.dim} x {self.input.dim} but it has {B.shape}"
        assert init_state.shape == (
            self.dim,
            1,
        ), f"init_state needs to have shape of {self.dim} x 1 but it has {init_state.shape}"

        self.A, self.B, self.init_state, self.bias = A, B, init_state, bias
        self._state = copy.deepcopy(init_state)

    @property
    def state(self) -> sympy.MatrixBase:
        return self._state

    def reset(self):
        self._state = copy.deepcopy(self.init_state)

    def forward(self, x: Matrix) -> Matrix:
        x = self.prepare_variable(x)

        assert x.shape == (
            self.input.dim,
            1,
        ), f"Input size mismatch: LinState layer expected {(self.input.dim,1)} but got {x.shape}"

        s = self.A @ self._state + self.B @ x + self.bias

        # we need to simplify here, otherwise we run the risk that states
        # will become increasingly more complex symbolic expressions
        if self._state.mode is MatrixMode.SYMBOLIC:
            try:
                self._state.matrix = sympy.simplify(s.matrix)
            except Exception as e:
                warnings.warn(f"Failed to simplify state {self.name}: {e}")
                self._state = s
        else:
            self._state = s

        return self._state


class Multiplicative(Variable):
    def __init__(self, input: Variable, name="Multiplicative"):

        if input.dim % 2 != 0:
            raise ValueError("Multiplicative layer requires an even number of inputs")

        super().__init__(dim=input.dim // 2, name=name)
        self.input = input
        self.size_str = f"{input.dim}/2->{self.dim}"

    def forward(self, x: Matrix) -> Matrix:
        x = self.prepare_variable(x)

        assert x.shape == (
            self.input.dim,
            1,
        ), f"Input size mismatch: Multiplicative layer expected {(self.input.dim,1)} but got {x.shape}"

        first_part = x[: self.dim, :]
        second_part = x[self.dim :, :]
        return Matrix.elementwise_multiplication(first_part, second_part)


def f_relu_identity(x: Variable) -> Variable:
    """Identity using ReLUs."""

    positive_part = ReLU(x)
    negative_part = ReLU(
        Linear(
            input=x,
            A=(-sympy.core.numbers.One()) * Matrix.eye(x.dim),
            b=Matrix.zeros(x.dim, 1),
            name="FlipSign",
        )
    )
    both = Concat([positive_part, negative_part])
    sum_component_wise = Linear(
        input=both,
        A=Matrix.hstack(Matrix.eye(x.dim), (-1 * sympy.core.numbers.One()) * Matrix.eye(x.dim)),
        b=Matrix.zeros(x.dim, 1),
        name="PosAndNegFuse",
    )
    return sum_component_wise
