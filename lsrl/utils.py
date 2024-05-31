import copy
import hashlib
from typing import Callable, List
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sympy
import warnings
import os

import sympy.matrices


def plot_and_save_graph(graph, file_path=None, figsize=(30, 30), node_types_instead_of_name=False):
    # Check if the graph is directed
    if not graph.is_directed():
        raise ValueError("The graph is not directed!")

    # create the directory if necessary and if a directory is present in the path
    if file_path is not None and not os.path.exists(os.path.dirname(file_path)) and os.path.dirname(file_path) != "":
        os.makedirs(os.path.dirname(file_path))

    # Attempt to plot the graph
    try:
        # # Position nodes using a layout (e.g., spring layout)
        # pos = nx.spring_layout(graph)

        for layer, nodes in enumerate(nx.topological_generations(graph)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                graph.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        pos = nx.multipartite_layout(graph, subset_key="layer", align="horizontal")
        # add some horizontal jitter to the node positions in order to avoid overlapping edges
        for key in pos:
            pos[key][0] += 0.02 * np.random.rand()  # Adjust the multiplier for more or less jiggle

        # ns_fun = (
        #     lambda node: f"{node.input_dim if hasattr(node, 'input_dim') else '?'}->{node.dim if hasattr(node, 'dim') else '?'}"
        # )
        # node_labels = {
        #     node: (
        #         f"{node.name}\n{ns_fun(node)}"
        #         if hasattr(node, "name")
        #         else "Unknown node"
        #     )
        #     for node in graph.nodes
        # }

        node_types = {node: node.__class__.__name__.split(".")[-1] for node in graph.nodes}
        color_map = {
            "Input": "#b2ff66",
            "Linear": "#66ffff",
            "Slice": "#66ffff",
            "ReLU": "#b266ff",
            "Concat": "#ffff33",
            "LinState": "#ff6666",
            "Multiplicative": "#ff66b2",
        }
        node_colors = [
            color_map[node_types[node]] if node_types[node] in color_map else "#e0e0e0" for node in graph.nodes
        ]

        # node_labels = {node: (f"{node.name if hasattr(node, 'name') else ''}\n{node.size_str}") for node in graph.nodes}
        if node_types_instead_of_name:
            node_labels = {node: node.__class__.__name__.split(".")[-1] for node in graph.nodes}
        else:
            node_labels = {node: f"{node.name if hasattr(node, 'name') else ''}" for node in graph.nodes}
            node_labels = {node: f"{name}\n{node.size_str}" for node, name in node_labels.items()}

        # Draw the graph using the positions
        plt.figure(figsize=figsize)
        nx.draw(
            graph,
            pos,
            # with_labels=True,
            labels=node_labels,
            node_color=node_colors,
            edge_color="k",
            node_size=1000,
            font_size=10,
            font_color="black",
            arrowstyle="-|>",
            arrowsize=10,
            # connectionstyle='arc3, rad = 0.1'
        )

        # Add labels to the edges (optional)
        # edge_labels = nx.get_edge_attributes(graph, 'weight')
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

        # Save the plot to a file
        if file_path is not None:
            plt.savefig(file_path)
            plt.close()  # Close the plot to free up memory

    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")


def is_directed_path_graph(G):
    # Check if the graph is empty
    if G.number_of_nodes() == 0:
        return False

    # Initialize degree counts
    start_vertex_count = 0
    end_vertex_count = 0
    middle_vertex_count = 0

    # Check the in-degree and out-degree of each vertex
    for node in G.nodes():
        out_degree = G.out_degree(node)
        in_degree = G.in_degree(node)

        if out_degree == 1 and in_degree == 0:
            start_vertex_count += 1
        elif out_degree == 0 and in_degree == 1:
            end_vertex_count += 1
        elif out_degree == 1 and in_degree == 1:
            middle_vertex_count += 1
        else:
            return False

    # Ensure there is exactly one start and one end vertex
    if start_vertex_count != 1 or end_vertex_count != 1:
        return False

    # Check if all other vertices are middle vertices
    if middle_vertex_count != G.number_of_nodes() - 2:
        return False

    # Check if the graph is weakly connected
    if not nx.is_weakly_connected(G):
        return False

    return True


def create_test_target_function(in_dim: int, out_dim: int, n_control_points: int, smoothness: float):

    # Generate n random control in_dim-dimensional points with support -0.1, 1.1
    control_points = np.random.rand(n_control_points, in_dim) * 1.2 - 0.1

    # Generate n random out_dim-dimensional output values with support -3 3
    control_values = np.random.rand(n_control_points, out_dim) * 6 - 3

    def target_function(x: np.array):
        # Compute the distance between x and each control point
        distances = np.linalg.norm(control_points - x, axis=1)

        # Compute the weights based on the distances
        weights = np.exp(-distances / smoothness)
        weights /= np.sum(weights)

        # Compute the output as a weighted sum of control values
        return np.sum(weights[:, None] * control_values, axis=0)

    return target_function


def construct_prompt(fun: Callable, in_dim: int, out_dim: int, N: int):
    # Discretize the [0,1]^in_dim domain into N steps and get the centers
    tick_centers = []
    for _ in range(in_dim):
        x = np.linspace(0, 1, N + 1)
        tick_centers.append((x[:-1] + x[1:]) / 2)

    # Create a matrix to store indices and function values
    result_matrix = np.zeros((N**in_dim, 1 + in_dim + out_dim))

    # Counter for the current row in the result matrix
    count = 0

    # Evaluate the function at each center of the domain
    for indices in itertools.product(*[enumerate(tc) for tc in tick_centers]):
        locs = [p[0] / N for p in indices]
        center_value = [p[1] for p in indices]

        values = fun(np.array(center_value))
        result_matrix[count, :] = [1 / N] + list(locs) + list(values)
        count += 1

    return result_matrix


def is_symbolic(x):
    return isinstance(x, (sympy.Basic, sympy.matrices.matrices.MatrixBase))


def is_fully_symbolic(x):
    return isinstance(x, (sympy.Basic, sympy.matrices.matrices.MatrixBase)) and len(x.atoms(sympy.Float)) == 0


def diag_matrices_and_eyes(matrices: List[sympy.SparseMatrix | sympy.Matrix | int]) -> sympy.SparseMatrix:
    """This is much faster than sympy.SparseMatrix.diag"""
    rows, cols = 0, 0
    elements = {}
    for m in matrices:
        # if dense sympy matrix put the whole matrix in the elements:
        if isinstance(m, sympy.SparseMatrix):
            elements.update({(rows + i, cols + j): m for (i, j), m in m.todok().items()})
            rows += m.rows
            cols += m.cols
        elif isinstance(m, sympy.Matrix):
            elements.update({(rows, cols): m})
            rows += m.rows
            cols += m.cols
        elif isinstance(m, int):
            elements.update({(rows + i, cols + i): sympy.core.numbers.One() for i in range(m)})
            rows += m
            cols += m
        else:
            raise ValueError(f"Invalid matrix type: {type(m)}")

    return sympy.SparseMatrix(rows, cols, elements)
