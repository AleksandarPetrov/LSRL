import copy
from itertools import product
import os
from typing import Literal
import warnings
import unittest

import numpy as np
import sympy
import lsrl
import multiprocessing
from parameterized import parameterized_class

import matplotlib.pyplot as plt

from lsrl.matrix_tools import Matrix, MatrixMode, matrix_mode


def run_instance(filename, x_vals, prompts):

    loop = lsrl.ForEach.load(filename)
    target_fun_output_dim = loop.input.dim - 1 - x_vals.shape[1]
    results = []
    for x_val in x_vals:
        first_input = np.concatenate((x_val, np.zeros(target_fun_output_dim + 1)))
        full_inputs = np.vstack((first_input, prompts))
        outputs = loop(full_inputs.T)
        results.append(outputs.numpy()[:, -1])
    return results


def run_parallel(filename: str, x_values, prompts):
    num_processes = min(multiprocessing.cpu_count() // 2, len(x_values))
    pool = multiprocessing.Pool(processes=num_processes)
    x_vals_split = np.array_split(x_values, num_processes)
    results = pool.starmap(run_instance, [(filename, x_vals, prompts) for x_vals in x_vals_split])
    pool.close()
    pool.join()
    return np.vstack(results).squeeze()


@parameterized_class(
    [
        {"mode": mode, "conditional_mode": conditional_mode, "name": f"{mode.name}, {conditional_mode}"}
        for mode in MatrixMode
        for conditional_mode in ["multiplicative", "direct", "optimized", "multiplicative_optimized", "step_optimized"]
    ]
)
class TestUAVec2Vec(unittest.TestCase):

    mode: MatrixMode
    conditional_mode: Literal["multiplicative", "direct", "optimized", "multiplicative_optimized", "step_optimized"]
    make_plots: bool = True

    def tearDown(self):
        self._mode_manager.__exit__(None, None, None)

    def setUp(self):

        self._mode_manager = matrix_mode(self.mode)
        self._mode_manager.__enter__()

        self.target_fun_input_dim = 2
        self.target_fun_output_dim = 2
        STEP_SCALE = 1_000

        # input dimension is such that we can express the
        # discretization size (1/N), the current cell location and the
        # output value of the self.target function at that cell
        input = lsrl.Input(dim=1 + self.target_fun_input_dim + self.target_fun_output_dim)

        # constant 1 that we will need for the counter
        const_1 = lsrl.f_constant(input, 1)

        # this counter gives us the current input number
        # we use it to differentiate between the input at position 1 and the self.prompts at positions > 1
        counter = lsrl.LinState(
            input=const_1,
            A=Matrix.eye(1),
            B=Matrix.eye(1),
            init_state=Matrix.zeros(1, 1),
            name="Counter",
        )

        # we have to up-project the counter and the const vector as we don't support broadcasting (yet)
        counter_vector = lsrl.Linear(
            input=counter,
            A=Matrix.ones(self.target_fun_input_dim, 1),
            b=Matrix.zeros(self.target_fun_input_dim, 1),
            name="CounterVector",
        )
        # const_1_vector_in = lsrl.Linear(
        #     input=const_1,
        #     A=Matrix.ones(self.target_fun_input_dim, 1),
        #     b=Matrix.zeros(self.target_fun_input_dim, 1),
        #     name="Const1Vector",
        # )

        # we want to keep the input in a state so we need to update this state only when the counter is 1
        # x_update = lsrl.f_ifeq(
        #     counter_vector,
        #     const_1_vector_in,
        #     input[: self.target_fun_input_dim],
        #     input[: self.target_fun_input_dim] * 0,
        #     scale=STEP_SCALE,
        # )
        large_constant = 1e10

        if self.conditional_mode == "direct":
            x_update = lsrl.f_ifelse(
                lsrl.f_smaller(counter_vector, 1.5),
                t_val=input[: self.target_fun_input_dim],
                f_val=input[: self.target_fun_input_dim] * 0,
            )
        elif self.conditional_mode == "step_optimized":
            x_update = lsrl.f_ifelse_step(
                lsrl.f_smaller(counter_vector, 1.5), t_val=input[: self.target_fun_input_dim], f_val=None
            )
        elif self.conditional_mode == "optimized":
            x_update = lsrl.ReLU(
                lsrl.f_larger(counter_vector, 1.5) * (-1 * large_constant) + input[: self.target_fun_input_dim]
            )
        elif self.conditional_mode == "multiplicative":
            x_update = lsrl.f_ifelse_mul(
                lsrl.f_smaller(counter_vector, 1.5),
                t_val=input[: self.target_fun_input_dim],
                f_val=input[: self.target_fun_input_dim] * 0,
            )
        elif self.conditional_mode == "multiplicative_optimized":
            x_update = lsrl.f_ifelse_mul(
                lsrl.f_smaller(counter_vector, 1.5), t_val=input[: self.target_fun_input_dim], f_val=None
            )
        else:
            raise ValueError(f"Invalid conditional mode {self.conditional_mode}")

        x = lsrl.LinState(
            input=x_update,
            A=Matrix.eye(self.target_fun_input_dim),
            B=Matrix.eye(self.target_fun_input_dim),
            init_state=Matrix.zeros(self.target_fun_input_dim, 1),
            name="x",
        )

        # now we can process the prompt inputs
        # note that we should ignore the first input as it is the x

        # we need the lower bound and the upper boundaries for the current cell
        # the lower bound is directly given in the prompt tokens
        # the upper bound we get by adding 1/N to the lower bound

        # broadcast the step size to the input dimension
        step_size = lsrl.Linear(
            input[0],
            Matrix.ones(self.target_fun_input_dim, 1),
            Matrix.zeros(self.target_fun_input_dim, 1),
            name="StepSizeBroadcast",
        )
        lb = input[1 : 1 + self.target_fun_input_dim]
        ub = lb + step_size

        # we can now create the bump function that checks if x is in this cell

        x_in_bump_componentwise = lsrl.f_bump(x, lb, ub, scale=STEP_SCALE)
        # this would be component-wise test so we need to check if all entries are 1
        bump_sum = lsrl.Linear(
            input=x_in_bump_componentwise,
            A=Matrix.ones(1, self.target_fun_input_dim),
            b=Matrix.zeros(1, 1),
            name="BumpSum",
        )

        # if the bump sum is equal to self.target_fun_input_dim we are in the cell
        # and if the counter is > 1 we are in the state of processing the self.prompts
        # so we can update the output
        in_cell = lsrl.f_larger(bump_sum, self.target_fun_input_dim - 0.5, scale=STEP_SCALE)
        in_cell_and_processing = lsrl.f_and(in_cell, lsrl.f_larger(counter, 0.5, scale=STEP_SCALE), scale=STEP_SCALE)

        # again, we need to up-project in_cell_and_processing to the output dimension as we do not support broadcasting
        in_cell_and_processing_vector = lsrl.Linear(
            input=in_cell_and_processing,
            A=Matrix.ones(self.target_fun_output_dim, 1),
            b=Matrix.zeros(self.target_fun_output_dim, 1),
            name="InCellAndProcessingVector",
        )
        const_1_vector_out = lsrl.Linear(
            input=const_1,
            A=Matrix.ones(self.target_fun_output_dim, 1),
            b=Matrix.zeros(self.target_fun_output_dim, 1),
            name="Const1Vector",
        )
        # update = lsrl.f_ifeq(
        #     in_cell_and_processing_vector,
        #     const_1_vector_out,
        #     input[self.target_fun_input_dim + 1 :],
        #     input[self.target_fun_input_dim + 1 :] * 0,
        #     scale=STEP_SCALE,
        # )

        if self.conditional_mode == "direct":
            update = lsrl.f_ifelse(
                lsrl.f_larger(in_cell_and_processing_vector, 0.5),
                t_val=input[self.target_fun_input_dim + 1 :],
                f_val=input[self.target_fun_input_dim + 1 :] * 0,
            )
        elif self.conditional_mode == "optimized":
            update = lsrl.ReLU(
                lsrl.f_smaller(in_cell_and_processing_vector, 0.5) * (-1 * large_constant)
                + input[self.target_fun_input_dim + 1 :]
            ) - lsrl.ReLU(
                lsrl.f_smaller(in_cell_and_processing_vector, 0.5) * (-1 * large_constant)
                - input[self.target_fun_input_dim + 1 :]
            )
        elif self.conditional_mode == "step_optimized":
            update = lsrl.f_ifelse_step(
                lsrl.f_larger(in_cell_and_processing_vector, 0.5),
                t_val=input[self.target_fun_input_dim + 1 :],
                f_val=None,
            )
        elif self.conditional_mode == "multiplicative":
            update = lsrl.f_ifelse_mul(
                lsrl.f_larger(in_cell_and_processing_vector, 0.5),
                t_val=input[self.target_fun_input_dim + 1 :],
                f_val=input[self.target_fun_input_dim + 1 :] * 0,
            )
        elif self.conditional_mode == "multiplicative_optimized":
            update = lsrl.f_ifelse_mul(
                lsrl.f_larger(in_cell_and_processing_vector, 0.5),
                t_val=input[self.target_fun_input_dim + 1 :],
                f_val=None,
            )
        else:
            raise ValueError(f"Invalid conditional mode {self.conditional_mode}")

        # we can now update the output
        y = lsrl.LinState(
            input=update,
            A=Matrix.eye(self.target_fun_output_dim),
            B=Matrix.eye(self.target_fun_output_dim),
            init_state=Matrix.zeros(self.target_fun_output_dim, 1),
            name="Output",
        )

        self.loop = lsrl.ForEach(output=y)

        ### SETUP THE TEST DATA

        # Create the test target function
        self.target = lsrl.utils.create_test_target_function(
            in_dim=self.target_fun_input_dim,
            out_dim=self.target_fun_output_dim,
            n_control_points=5,
            smoothness=1,
        )

        # plot the self.target function if the input dim is 2 with a separate plot for each output dimension
        if not os.path.exists("test_results"):
            os.makedirs("test_results")

        if self.target_fun_input_dim == 2 and self.make_plots:
            x = np.linspace(0, 1, 100)
            y = np.linspace(0, 1, 100)
            x_centers = (x[:-1] + x[1:]) / 2
            y_centers = (y[:-1] + y[1:]) / 2
            X, Y = np.meshgrid(x_centers, y_centers)
            Z = np.zeros((len(x_centers), len(y_centers), self.target_fun_output_dim))
            for i, self.x_val in enumerate(x_centers):
                for j, y_val in enumerate(y_centers):
                    Z[i, j] = self.target([self.x_val, y_val])
            for i in range(self.target_fun_output_dim):
                fig = plt.figure()
                plt.imshow(Z[:, :, i])
                plt.colorbar()
                plt.savefig(f"test_results/ua_vec2vec_target_function_output_{i}.png")
        # construct the approximation prompt
        discretization_levels = 4

        self.prompts = lsrl.utils.construct_prompt(
            self.target,
            in_dim=self.target_fun_input_dim,
            out_dim=self.target_fun_output_dim,
            N=discretization_levels,
        )

        self.test_x_vals = np.random.random((20, self.target_fun_input_dim))
        self.test_y_vals = np.array([self.target(x) for x in self.test_x_vals])

        if not os.path.exists("test_results"):
            os.makedirs("test_results")

        if self.make_plots:
            lsrl.utils.plot_and_save_graph(self.loop.graph(), f"test_results/ua_vec2vec_before_simplification.png")
            print(f"Number of nodes before: {self.loop.graph().number_of_nodes()}")
            print(
                f"Number of nodes with out-degree more than 1: {len([n for n in self.loop.graph().nodes() if self.loop.graph().out_degree(n) > 1])}"
            )

        ### PREPARE THE SIMPLIFIED MODEL, WE WILL LOAD IT FROM
        ### CACHE IF IT EXISTS AS SIMPLIFICATION IS VERY SLOW
        self.original_filename = f"test_results/ua_vec2vec_original_{self.mode.name}_{self.conditional_mode}.model"
        self.loop.save(self.original_filename)

        # self.simplified_filename = f"test_results/ua_vec2vec_simplified_{self.mode.name}_NOISY.model"
        self.simplified_filename = f"test_results/ua_vec2vec_simplified_{self.mode.name}_{self.conditional_mode}.model"

        # if exists print warning that we won't compute new simplified
        if os.path.exists(self.simplified_filename):
            warnings.warn(f"File {self.simplified_filename} exists. NOT COMPUTING NEW SIMPLIFIED MODEL.", UserWarning)
            self.simplified = lsrl.ForEach.load(self.simplified_filename)
        else:
            self.simplified = copy.deepcopy(self.loop)
            self.simplified.simplify(max_steps=2000)

            self.simplified.save(self.simplified_filename)

    def approximation_test(self, test_name: str, simplification: bool, x: np.array, y: np.array):

        if simplification:
            filename = self.simplified_filename
        else:
            filename = self.original_filename

        assert self.target_fun_input_dim == 2

        x_centers = (x[:-1] + x[1:]) / 2
        y_centers = (y[:-1] + y[1:]) / 2
        Z = np.zeros((len(x_centers), len(y_centers), self.target_fun_output_dim))
        Z_target = np.zeros((len(x_centers), len(y_centers), self.target_fun_output_dim))

        # process the inputs through the model
        ij_x_vals = np.array(
            [[i, j, x_val, y_val] for (i, x_val), (j, y_val) in product(enumerate(x_centers), enumerate(y_centers))]
        )
        results = run_parallel(filename, ij_x_vals[:, -2:], self.prompts)

        for loc_x_y, res in zip(ij_x_vals, results):
            i, j, x_val, y_val = loc_x_y
            Z[int(i), int(j)] = res
            Z_target[int(i), int(j)] = self.target([x_val, y_val])

        for i in range(self.target_fun_output_dim):
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            v_min = min(np.min(Z[:, :, i]), np.min(Z_target[:, :, i]))
            v_max = max(np.max(Z[:, :, i]), np.max(Z_target[:, :, i]))

            # Plot Z
            axs[0].imshow(Z[:, :, i], vmin=v_min, vmax=v_max, aspect="equal")
            axs[0].set_title("Model Output")
            axs[0].set_xlabel("X")
            axs[0].set_ylabel("Y")

            # Plot Z_target
            im2 = axs[1].imshow(Z_target[:, :, i], vmin=v_min, vmax=v_max, aspect="equal")
            axs[1].set_title("Target Output")
            axs[1].set_xlabel("X")
            axs[1].set_ylabel("Y")
            fig.colorbar(im2, ax=axs[1])

            plt.savefig(f"test_results/ua_vec2vec_model_vs_target_{test_name}_{i}.png")

        for i, j in product(range(len(x_centers)), range(len(y_centers))):
            dist = np.abs(Z[i, j] - Z_target[i, j]).flatten().max()
            print(
                f"{i},{j} max_diff={dist:.4f}. \t{'Simplified' if simplification else 'Original'} model: {Z[i, j]}, \tTarget function: {Z_target[i, j]}"
            )

        np.testing.assert_allclose(Z, Z_target, atol=1e-1, rtol=1e-1, verbose=True)

    APPROXIMATION_TEST_RESOLUTION = 5

    def test_approximation_before_simplification_coarse(self):
        """Checks how close the approximation is to the target function before simplificaiton on a coarse grid."""
        self.approximation_test(
            "Before simp. coarse",
            simplification=False,
            x=np.linspace(0, 1, self.APPROXIMATION_TEST_RESOLUTION),
            y=np.linspace(0, 1, self.APPROXIMATION_TEST_RESOLUTION),
        )

    def test_approximation_before_simplification_fine(self):
        """Checks how close the approximation is to the target function before simplificaiton on a fine grid."""
        self.approximation_test(
            "Before simp. fine",
            simplification=False,
            x=np.linspace(0.1, 0.2, self.APPROXIMATION_TEST_RESOLUTION),
            y=np.linspace(0.1, 0.2, self.APPROXIMATION_TEST_RESOLUTION),
        )

    def test_approximation_after_simplification_coarse(self):
        """Checks how close the approximation is to the target function after simplificaiton on a coarse grid."""
        self.approximation_test(
            "After simp. coarse",
            simplification=True,
            x=np.linspace(0, 1, self.APPROXIMATION_TEST_RESOLUTION),
            y=np.linspace(0, 1, self.APPROXIMATION_TEST_RESOLUTION),
        )

    def test_approximation_after_simplification_fine(self):
        """Checks how close the approximation is to the target function after simplificaiton on a fine grid."""
        self.approximation_test(
            "After simp. fine",
            simplification=True,
            x=np.linspace(0.1, 0.2, self.APPROXIMATION_TEST_RESOLUTION),
            y=np.linspace(0.1, 0.2, self.APPROXIMATION_TEST_RESOLUTION),
        )

    def test_simplification(self):
        """Compares model outputs before and after simplificaiton."""

        print(self.loop(np.vstack((np.array([0.1, 0.1, 0, 0, 0]), self.prompts)).T).numpy().T)
        print(self.simplified(np.vstack((np.array([0.1, 0.1, 0, 0, 0]), self.prompts)).T).numpy().T)

        results_original = run_parallel(self.original_filename, self.test_x_vals, self.prompts)
        results_simplified = run_parallel(self.simplified_filename, self.test_x_vals, self.prompts)

        for i in range(len(self.test_x_vals)):
            if not np.allclose(results_original[i], results_simplified[i], atol=1e-4, rtol=1e-4):
                print(f"Original: {results_original[i]} Simplified: {results_simplified[i]}")

        np.testing.assert_allclose(results_original, results_simplified, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
