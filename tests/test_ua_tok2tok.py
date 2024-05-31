from collections import deque
import copy
from itertools import cycle, islice, product
import os
import tempfile
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
from lsrl.utils import is_directed_path_graph


@parameterized_class(
    [
        {"mode": mode, "conditional_mode": conditional_mode, "name": f"{mode.name}, {conditional_mode}"}
        for mode in MatrixMode
        for conditional_mode in ["multiplicative", "direct", "optimized", "step_optimized", "multiplicative_optimized"]
    ]
)
class TestUATok2TokBuldingSteps(unittest.TestCase):

    vocab_size = 10
    n_samples = 20
    seq_len = 3

    mode: MatrixMode
    conditional_mode: Literal["multiplicative", "direct", "optimized", "step_optimized", "multiplicative_optimized"]

    def tearDown(self):
        self._mode_manager.__exit__(None, None, None)

    def setUp(self, mode=None, mul=None):

        if mode is not None:
            self.mode = mode

        if mul is not None:
            self.mul = mul

        self._mode_manager = matrix_mode(self.mode)
        self._mode_manager.__enter__()

        STEP_SCALE = 1_000

        # CREATE THE MODEL
        input = lsrl.Input(dim=1)
        input.always_positive = True

        const_0 = lsrl.f_constant(input, 0)
        const_0.always_positive = True
        const_1 = lsrl.f_constant(input, 1)
        const_1.always_positive = True

        # this counter starts at 0
        global_counter = lsrl.LinState(
            input=const_1,
            A=Matrix.eye(1),
            B=Matrix.eye(1),
            init_state=Matrix([[-1]]),
            name="GlobalCounter",
        )
        global_counter.always_positive = True

        modT_counter = lsrl.f_modulo_counter(input, self.seq_len)
        mod2T_counter = lsrl.f_modulo_counter(input, 2 * self.seq_len)

        self.debug_counters = lsrl.Concat([global_counter, modT_counter, mod2T_counter])

        # adding the 0.5 for numerical stability
        is_prompt = lsrl.f_larger(global_counter, self.seq_len - 0.5, scale=STEP_SCALE)
        is_compare_mode = lsrl.f_larger(mod2T_counter, self.seq_len - 0.5, scale=STEP_SCALE)
        is_copy_mode = lsrl.f_and(
            is_prompt,
            lsrl.f_not(is_compare_mode),
            scale=STEP_SCALE,
        )
        is_first_token_for_copy = lsrl.f_and(
            is_copy_mode,
            lsrl.f_smaller(modT_counter, 0.5, scale=STEP_SCALE),
            scale=STEP_SCALE,
        )

        self.debug_switches = lsrl.Concat(
            [global_counter, is_prompt, is_compare_mode, is_copy_mode, is_first_token_for_copy]
        )

        # tmp var to hold the update to the state that keeps the input
        # t_query_update_single = lsrl.f_ifeq(is_prompt, const_0, input, const_0, scale=STEP_SCALE)
        large_constant = 1e10
        if self.conditional_mode == "direct":
            t_query_update_single = lsrl.f_ifelse(lsrl.f_smaller(is_prompt, 0.5), t_val=input, f_val=input * 0)
        elif self.conditional_mode == "step_optimized":
            t_query_update_single = lsrl.f_ifelse_step(lsrl.f_smaller(is_prompt, 0.5), t_val=input, f_val=None)
        elif self.conditional_mode == "optimized":
            t_query_update_single = lsrl.ReLU(lsrl.f_larger(is_prompt, 0.5) * (-1 * large_constant) + input)
        elif self.conditional_mode == "multiplicative":
            t_query_update_single = lsrl.f_ifelse_mul(lsrl.f_smaller(is_prompt, 0.5), t_val=input, f_val=input * 0)
        elif self.conditional_mode == "multiplicative_optimized":
            t_query_update_single = lsrl.f_ifelse_mul(lsrl.f_smaller(is_prompt, 0.5), t_val=input, f_val=None)
        else:
            raise ValueError(f"Invalid conditional mode {self.conditional_mode}")

        t_query_update_single.always_positive = True

        t_query_update_whole_seq = []
        for i in range(self.seq_len):  # this creates self.seq_len individual operations

            if self.conditional_mode == "direct":
                t_query_update_whole_seq.append(
                    lsrl.f_ifelse(
                        lsrl.f_and(lsrl.f_larger(modT_counter, i - 0.5), lsrl.f_smaller(modT_counter, i + 0.5)),
                        t_val=t_query_update_single,
                        f_val=t_query_update_single * 0,
                    )
                )
            elif self.conditional_mode == "step_optimized":
                t_query_update_whole_seq.append(
                    lsrl.f_ifelse_step(
                        lsrl.f_and(lsrl.f_larger(modT_counter, i - 0.5), lsrl.f_smaller(modT_counter, i + 0.5)),
                        t_val=t_query_update_single,
                        f_val=None,
                    )
                )
            elif self.conditional_mode == "optimized":
                t_query_update_whole_seq.append(
                    lsrl.ReLU(
                        lsrl.f_or(
                            lsrl.f_larger(modT_counter, i + 0.5, scale=STEP_SCALE),
                            lsrl.f_smaller(modT_counter, i - 0.5, scale=STEP_SCALE),
                        )
                        * (-1 * large_constant)
                        + t_query_update_single
                    )
                    # lsrl.f_ifeq(modT_counter, lsrl.f_constant(input, i), t_query_update_single, const_0, scale=STEP_SCALE)
                )
            elif self.conditional_mode == "multiplicative":
                t_query_update_whole_seq.append(
                    lsrl.f_ifelse_mul(
                        lsrl.f_and(lsrl.f_larger(modT_counter, i - 0.5), lsrl.f_smaller(modT_counter, i + 0.5)),
                        t_val=t_query_update_single,
                        f_val=t_query_update_single * 0,
                    )
                )
            elif self.conditional_mode == "multiplicative_optimized":
                t_query_update_whole_seq.append(
                    lsrl.f_ifelse_mul(
                        lsrl.f_and(lsrl.f_larger(modT_counter, i - 0.5), lsrl.f_smaller(modT_counter, i + 0.5)),
                        t_val=t_query_update_single,
                        f_val=None,
                    )
                )
            else:
                raise ValueError(f"Invalid conditional mode {self.conditional_mode}")

        query = lsrl.LinState(
            input=lsrl.Concat(t_query_update_whole_seq),
            A=Matrix.eye(self.seq_len),
            B=Matrix.eye(self.seq_len),
            init_state=Matrix.zeros(self.seq_len, 1),
            name="Query",
        )
        query.always_positive = True

        self.debug_query = query

        # if we are in compare mode check if the current
        # element matches the one in the same position in the query
        corresponding_query_element_whole_seq = []
        for i in range(self.seq_len):  # this creates self.seq_len individual operations

            if self.conditional_mode == "direct":
                corresponding_query_element_whole_seq.append(
                    lsrl.f_ifelse(
                        lsrl.f_and(
                            lsrl.f_larger(modT_counter, i - 0.5, scale=STEP_SCALE),
                            lsrl.f_smaller(modT_counter, i + 0.5, scale=STEP_SCALE),
                        ),
                        t_val=query[i],
                        f_val=query[i] * 0,
                    )
                )
            elif self.conditional_mode == "step_optimized":
                corresponding_query_element_whole_seq.append(
                    lsrl.f_ifelse_step(
                        lsrl.f_and(
                            lsrl.f_larger(modT_counter, i - 0.5, scale=STEP_SCALE),
                            lsrl.f_smaller(modT_counter, i + 0.5, scale=STEP_SCALE),
                        ),
                        t_val=query[i],
                        f_val=None,
                    )
                )
            elif self.conditional_mode == "optimized":
                corresponding_query_element_whole_seq.append(
                    lsrl.ReLU(
                        lsrl.f_or(
                            lsrl.f_larger(modT_counter, i + 0.5, scale=STEP_SCALE),
                            lsrl.f_smaller(modT_counter, i - 0.5, scale=STEP_SCALE),
                        )
                        * (-1 * large_constant)
                        + query[i],
                    )
                )
            elif self.conditional_mode == "multiplicative":
                corresponding_query_element_whole_seq.append(
                    lsrl.f_ifelse_mul(
                        lsrl.f_and(
                            lsrl.f_larger(modT_counter, i - 0.5, scale=STEP_SCALE),
                            lsrl.f_smaller(modT_counter, i + 0.5, scale=STEP_SCALE),
                        ),
                        t_val=query[i],
                        f_val=query[i] * 0,
                    )
                )
            elif self.conditional_mode == "multiplicative_optimized":
                corresponding_query_element_whole_seq.append(
                    lsrl.f_ifelse_mul(
                        lsrl.f_and(
                            lsrl.f_larger(modT_counter, i - 0.5, scale=STEP_SCALE),
                            lsrl.f_smaller(modT_counter, i + 0.5, scale=STEP_SCALE),
                        ),
                        t_val=query[i],
                        f_val=None,
                    )
                )
            else:
                raise ValueError(f"Invalid conditional mode {self.conditional_mode}")

        # sum the elements (only one should be no-zero)
        corresponding_query_element = lsrl.Linear(
            input=lsrl.Concat(corresponding_query_element_whole_seq),
            A=Matrix.ones(1, self.seq_len),
            b=Matrix.zeros(1, 1),
        )
        corresponding_query_element.always_positive = True

        matching = lsrl.f_and(
            x=lsrl.f_and(
                lsrl.f_larger(input, corresponding_query_element - 0.5, scale=STEP_SCALE),
                lsrl.f_smaller(input, corresponding_query_element + 0.5, scale=STEP_SCALE),
            ),
            y=is_compare_mode,
            scale=STEP_SCALE,
        )

        # matching = lsrl.f_and(
        #     x=lsrl.f_ifeq(input, corresponding_query_element, const_1, const_0, scale=STEP_SCALE),
        #     y=is_compare_mode,
        #     scale=STEP_SCALE,
        # )
        matching.always_positive = True

        # keep a buffer of the last last T+1 values to check if we have a whole sequence match
        # the +1 is because we can only read from the buffer after we write to it

        shift_matrix = sympy.SparseMatrix.zeros(self.seq_len + 1, self.seq_len + 1)
        for i in range(self.seq_len):
            shift_matrix[i, i + 1] = 1
        shift_matrix = Matrix(shift_matrix)

        buffer = lsrl.LinState(
            input=matching,
            A=shift_matrix,
            B=Matrix([[0] for _ in range(self.seq_len)] + [[1]]),
            init_state=Matrix.zeros(self.seq_len + 1, 1),
            name="Buffer",
        )
        buffer.always_positive = True

        # if the first self.seq_len intems in the buffer are matching, then we have our whole sequence match
        buffer_sum = lsrl.Linear(
            input=buffer,
            A=Matrix([[1 for _ in range(self.seq_len)] + [0]]),
            b=Matrix.zeros(1, 1),
        )
        buffer_sum.always_positive = True
        all_matching = lsrl.f_larger(
            buffer_sum,
            self.seq_len - 0.5,
            scale=STEP_SCALE,
        )

        self.debug_matching = lsrl.Concat([matching, buffer, all_matching])

        # we have a state that designates when we have started copying
        # this would be zero until we have a complete match of the key
        # then it turns to the current counter number so that we can copy the
        # next seq_len tokens into a register as these are the values we need to
        # output at the end

        # t_started_on_update = lsrl.f_ifeq(
        #     x=lsrl.f_and(all_matching, is_first_token_for_copy, scale=STEP_SCALE),
        #     y=const_1,
        #     t_val=global_counter,
        #     f_val=const_0,
        #     scale=STEP_SCALE,
        # )

        matching_and_first_for_copy = lsrl.f_and(all_matching, is_first_token_for_copy, scale=STEP_SCALE)
        matching_and_first_for_copy.always_positive = True

        if self.conditional_mode == "direct":
            t_started_on_update = lsrl.f_ifelse(
                matching_and_first_for_copy, t_val=global_counter, f_val=global_counter * 0
            )
        elif self.conditional_mode == "step_optimized":
            t_started_on_update = lsrl.f_ifelse_step(matching_and_first_for_copy, t_val=global_counter, f_val=None)
        elif self.conditional_mode == "optimized":
            t_started_on_update = lsrl.ReLU(
                lsrl.f_not(matching_and_first_for_copy) * (-1 * large_constant) + global_counter
            )
        elif self.conditional_mode == "multiplicative":
            t_started_on_update = lsrl.f_ifelse_mul(
                matching_and_first_for_copy, t_val=global_counter, f_val=global_counter * 0
            )
        elif self.conditional_mode == "multiplicative_optimized":
            t_started_on_update = lsrl.f_ifelse_mul(matching_and_first_for_copy, t_val=global_counter, f_val=None)
        else:
            raise ValueError(f"Invalid conditional mode {self.conditional_mode}")

        started_on = lsrl.LinState(
            input=t_started_on_update,
            A=Matrix.eye(1),
            B=Matrix.eye(1),
            init_state=Matrix.zeros(1, 1),
            name="Started on",
        )
        started_on.always_positive = True

        # this variable designates whether we should be copying into the output register
        # should be true for seq_len steps after started_on
        is_copying_on = lsrl.f_smaller(global_counter, started_on + self.seq_len, scale=STEP_SCALE)

        # if we are copying, copy the current value in the corresponding position in the register
        # the register has self.seq_len Linear States, one for each output
        copy_and_on = lsrl.f_and(is_copy_mode, is_copying_on, scale=STEP_SCALE)
        # t_register_updates_should_update = [
        #     lsrl.f_and(
        #         copy_and_on,
        #         lsrl.f_ifeq(
        #             x=modT_counter,
        #             y=lsrl.f_constant(copy_and_on, i),
        #             t_val=const_1,
        #             f_val=const_0,
        #             scale=STEP_SCALE,
        #         ),
        #     )
        #     for i in range(self.seq_len)
        # ]
        modT_counter_equal_to_i = [
            lsrl.f_and(
                lsrl.f_larger(modT_counter, i - 0.5, scale=STEP_SCALE),
                lsrl.f_smaller(modT_counter, i + 0.5, scale=STEP_SCALE),
                scale=STEP_SCALE,
            )
            for i in range(self.seq_len)
        ]
        t_register_updates_should_update = [
            lsrl.f_and(x=copy_and_on, y=modT_counter_equal_to_i[i], scale=STEP_SCALE) for i in range(self.seq_len)
        ]

        if self.conditional_mode == "direct":
            t_register_updates = [
                lsrl.f_ifelse(lsrl.f_larger(t_register_updates_should_update[i], 0.5), t_val=input, f_val=input * 0)
                for i in range(self.seq_len)
            ]
        elif self.conditional_mode == "step_optimized":
            t_register_updates = [
                lsrl.f_ifelse_step(lsrl.f_larger(t_register_updates_should_update[i], 0.5), t_val=input, f_val=None)
                for i in range(self.seq_len)
            ]
        elif self.conditional_mode == "optimized":
            t_register_updates = [
                lsrl.ReLU(lsrl.f_smaller(t_register_updates_should_update[i], 0.5) * (-1 * large_constant) + input)
                for i in range(self.seq_len)
            ]
        elif self.conditional_mode == "multiplicative":
            t_register_updates = [
                lsrl.f_ifelse_mul(lsrl.f_larger(t_register_updates_should_update[i], 0.5), t_val=input, f_val=input * 0)
                for i in range(self.seq_len)
            ]
        elif self.conditional_mode == "multiplicative_optimized":
            t_register_updates = [
                lsrl.f_ifelse_mul(lsrl.f_larger(t_register_updates_should_update[i], 0.5), t_val=input, f_val=None)
                for i in range(self.seq_len)
            ]
        else:
            raise ValueError(f"Invalid conditional mode {self.conditional_mode}")

        output_registers = [
            lsrl.LinState(
                input=update,
                A=Matrix.eye(1),
                B=Matrix.eye(1),
                init_state=Matrix.zeros(1, 1),
            )
            for update in t_register_updates
        ]
        for output_register in output_registers:
            output_register.always_positive = True

        self.debug_copying = lsrl.Concat([started_on, copy_and_on] + output_registers)

        # The only thing left is to output the corrsponding values from each register
        # we only care about the last seq_len outputs but nothing stops us from just
        # repeating the output one we know what it should be

        if self.conditional_mode == "direct":
            t_output_individual_updates = [
                lsrl.f_ifelse(
                    lsrl.f_larger(modT_counter_equal_to_i[i], 0.5),
                    t_val=output_registers[i],
                    f_val=output_registers[i] * 0,
                )
                for i in range(self.seq_len)
            ]
        elif self.conditional_mode == "step_optimized":
            t_output_individual_updates = [
                lsrl.f_ifelse_step(
                    lsrl.f_larger(modT_counter_equal_to_i[i], 0.5), t_val=output_registers[i], f_val=None
                )
                for i in range(self.seq_len)
            ]
        elif self.conditional_mode == "optimized":
            t_output_individual_updates = [
                lsrl.ReLU(lsrl.f_smaller(modT_counter_equal_to_i[i], 0.5) * (-1 * large_constant) + output_registers[i])
                for i in range(self.seq_len)
            ]
        elif self.conditional_mode == "multiplicative":
            t_output_individual_updates = [
                lsrl.f_ifelse_mul(
                    lsrl.f_larger(modT_counter_equal_to_i[i], 0.5),
                    t_val=output_registers[i],
                    f_val=output_registers[i] * 0,
                )
                for i in range(self.seq_len)
            ]
        elif self.conditional_mode == "multiplicative_optimized":
            t_output_individual_updates = [
                lsrl.f_ifelse_mul(lsrl.f_larger(modT_counter_equal_to_i[i], 0.5), t_val=output_registers[i], f_val=None)
                for i in range(self.seq_len)
            ]
        else:
            raise ValueError(f"Invalid conditional mode {self.conditional_mode}")

        # sum the individual updates
        self.output = lsrl.Linear(
            input=lsrl.Concat(t_output_individual_updates),
            A=Matrix.ones(1, self.seq_len),
            b=Matrix.zeros(1, 1),
        )

        # CREATE TEST DATA
        self.target_map = dict()

        # we sample keys and targets that are self.seq_len long sequences of ints between 0 and vocab_size-1
        while len(self.target_map) < self.n_samples:
            self.target_map[tuple(np.random.randint(0, self.vocab_size, self.seq_len))] = tuple(
                np.random.randint(0, self.vocab_size, self.seq_len)
            )

        # construct the prompt, that is sequence of (x,y) pairs
        prompt_concat_pairs = [list(xx) + list(yy) for xx, yy in self.target_map.items()]
        self.prompt = [item for sublist in prompt_concat_pairs for item in sublist]

        # if not mul ensure that no Multiplicative units are present in the model
        if self.conditional_mode not in ["multiplicative", "multiplicative_optimized"]:
            for node in lsrl.ForEach(self.output).topological_sort():
                self.assertNotIsInstance(node, lsrl.Multiplicative)

    def prep_random_input(self):
        # sample a random pair from the target map
        keys = list(self.target_map.keys())
        x = keys[np.random.choice(len(keys))]
        y = self.target_map[x]
        full_input = np.concatenate((x, self.prompt))
        return x, y, full_input

    def test_counters(self):
        """Check that the global and the modulo counters work as expected."""

        x, y, full_input = self.prep_random_input()
        output = lsrl.ForEach(self.debug_counters)(full_input[None, :])
        expected_1 = np.arange(len(full_input))
        expected_2 = list(islice(cycle(range(self.seq_len)), len(full_input)))
        expected_3 = list(islice(cycle(range(2 * self.seq_len)), len(full_input)))
        np.testing.assert_allclose(output.numpy(), np.vstack((expected_1, expected_2, expected_3)))

    def test_switches(self):
        """Tests that the mode selectors work as expected."""

        x, y, full_input = self.prep_random_input()
        output = lsrl.ForEach(self.debug_switches)(full_input[None, :])

        counter = np.arange(len(full_input))
        is_prompt = np.array([0 for _ in range(self.seq_len)] + [1 for _ in range(len(full_input) - self.seq_len)])
        is_compare_mode = [0 for _ in range(self.seq_len)]
        is_copy_mode = [0 for _ in range(self.seq_len)]

        for _ in range(len(self.target_map)):
            is_compare_mode += [1 for _ in range(self.seq_len)] + [0 for _ in range(self.seq_len)]
            is_copy_mode += [0 for _ in range(self.seq_len)] + [1 for _ in range(self.seq_len)]

        is_first_token_for_copy = [
            int(bool(is_prompt[i]) and not bool(is_compare_mode[i]) and counter[i] % self.seq_len == 0)
            for i in range(len(full_input))
        ]

        expected = np.vstack((counter, is_prompt, is_compare_mode, is_copy_mode, is_first_token_for_copy))
        np.testing.assert_allclose(output.numpy(), expected)

    def test_query(self):
        """Test if the state variable that is supposed to keep the user query x actually does it."""

        x, y, full_input = self.prep_random_input()
        output = lsrl.ForEach(self.debug_query)(full_input[None, :])

        # the query is supposed to be the input for the duration of the prompt,
        # except the first self.seq_len tokens where we are still accumulating it
        expected = np.tile(x, (len(full_input), 1)).T
        np.testing.assert_allclose(output.numpy()[:, self.seq_len :], expected[:, self.seq_len :])

    def test_matching(self):
        """Test if we correctly detect when we are looking at dictiionary keys and the corresponding value corresponds to the one in the query."""

        x, y, full_input = self.prep_random_input()
        output = lsrl.ForEach(self.debug_matching)(full_input[None, :])

        # the matching should be 0 for the first self.seq_len tokens
        # then for the next self.seq_len it is the same if equal to the corresponding element of x
        # then for the next self.seq_len it is 0 and then we repeat
        matching = np.zeros(len(full_input))
        all_matching = np.zeros(len(full_input))

        # create a fixed size fifo buffer of length self.seq_len+1 filled with zeros initially
        buffer_deque = deque(np.zeros(self.seq_len + 1), maxlen=self.seq_len + 1)

        buffer = np.zeros((len(full_input), self.seq_len + 1))
        for i in range(len(full_input)):
            if i % (2 * self.seq_len) >= self.seq_len:
                matching[i] = 1 if full_input[i] == x[i % self.seq_len] else 0

            buffer_deque.append(matching[i])
            buffer[i] = np.array(buffer_deque)
            if sum(buffer[i, : self.seq_len]) == self.seq_len:
                all_matching[i] = 1

        np.testing.assert_allclose(output.numpy(), np.hstack((matching[:, None], buffer, all_matching[:, None])).T)

    def test_copying(self):
        """Test if we correctly copy the values from the input to the output registers."""

        x, y, full_input = self.prep_random_input()
        output = lsrl.ForEach(self.debug_copying)(full_input[None, :])

        # we have a register of length self.seq_len that we fill with the values from the input
        # when we have a full match of the key
        started_on = np.zeros(len(full_input))
        copy_and_on = np.zeros(len(full_input))
        output_registers = np.zeros((len(full_input), self.seq_len))

        # divide the prompt into chunks of size 2*seq_len
        chunks = [tuple(self.prompt[i : i + 2 * self.seq_len]) for i in range(0, len(self.prompt), 2 * self.seq_len)]
        # find the chunk that corresponds to our x and y
        chunk_idx = chunks.index(x + y)
        # now this gives us the started on
        started_on[(chunk_idx + 1) * 2 * self.seq_len :] = (chunk_idx + 1) * 2 * self.seq_len
        # and is copying on is 1 for seq_len steps afte
        copy_and_on[(chunk_idx + 1) * 2 * self.seq_len : (chunk_idx + 1) * 2 * self.seq_len + self.seq_len] = 1

        for i in range(self.seq_len):
            output_registers[(chunk_idx + 1) * 2 * self.seq_len + i :, i] = y[i]

        np.testing.assert_allclose(
            output.numpy(),
            np.hstack((started_on[:, None], copy_and_on[:, None], output_registers)).T,
        )

    def test_output_is_correct(self):
        """
        Test if the output is correct.
        """
        loop = lsrl.ForEach(self.output)

        # save to a tmp file using the tmp library
        with tempfile.NamedTemporaryFile(delete=True) as f:
            filename = f.name
            loop.save(filename)

            errors = []
            expected = []
            actual = []
            full_inputs = []

            while len(expected) < 10:
                x, y, full_input = TestUATok2TokBuldingSteps.prep_random_input(self)

                # prevent repetitions
                if any([(x == fi[: len(x)]).all() for fi in full_inputs]):
                    continue

                full_inputs.append(full_input)
                expected.append(y)

            num_processes = min(multiprocessing.cpu_count() // 2, len(full_inputs))
            pool = multiprocessing.Pool(processes=num_processes)
            full_inputs_split = np.array_split(full_inputs, num_processes)
            results = pool.starmap(run_instance, [(filename, inpts, self.seq_len) for inpts in full_inputs_split])
            pool.close()
            pool.join()
            results = np.array(results).squeeze()

            for r in results:
                actual.append(r)
                errors.append((r != y).sum())

            for exp, act in zip(expected, results):
                print(f"expected:{tuple(exp)} actual:{tuple(act)}")

            self.assertTrue(np.sum((np.array(results).squeeze() - expected) != 0) == 0)


# @parameterized_class([{"mode": mode, "name": {mode.name}} for mode in MatrixMode])


@parameterized_class(
    [
        {"mode": mode, "conditional_mode": conditional_mode, "name": f"{mode.name}, {conditional_mode}"}
        for mode in [MatrixMode.NUMERIC]
        for conditional_mode in ["multiplicative", "direct", "optimized", "multiplicative_optimized"]
    ]
)
class TestUATok2TokSimplification(unittest.TestCase):

    vocab_size = 10
    n_samples = 20
    seq_len = 3
    mode: MatrixMode
    conditional_mode: Literal["multiplicative", "direct", "optimized", "multiplicative_optimized"]
    make_plots: bool = True
    no_save: bool = False

    def tearDown(self):
        self._mode_manager.__exit__(None, None, None)

    def setUp(self) -> None:
        # Use TestUATok2Tok to build the model and the test example
        TestUATok2TokBuldingSteps.setUp(self)

        self.loop = lsrl.ForEach(self.output)

        ### PREPARE THE SIMPLIFIED MODEL, WE WILL LOAD IT FROM
        ### CACHE IF IT EXISTS AS SIMPLIFICATION IS VERY SLOW
        # create the directory if necessary and if a directory is present in the path
        if not os.path.exists("test_results"):
            os.makedirs("test_results")

        self.original_filename = f"test_results/ua_tok2tok_original_{self.seq_len}_{self.n_samples}_{self.vocab_size}_{self.mode.name}_{self.conditional_mode}.model"
        if not self.no_save and not os.path.exists(self.original_filename):
            self.loop.save(self.original_filename)

        self.simplified_filename = f"test_results/ua_tok2tok_simplified_{self.seq_len}_{self.n_samples}_{self.vocab_size}_{self.mode.name}_{self.conditional_mode}.model"
        # if exists print warning that we won't compute new simplified
        if os.path.exists(self.simplified_filename):
            warnings.warn(f"File {self.simplified_filename} exists. NOT COMPUTING NEW SIMPLIFIED MODEL.", UserWarning)
            self.simplified = lsrl.ForEach.load(self.simplified_filename)
        else:
            self.simplified = copy.deepcopy(self.loop)

        self.simplified.simplify(max_steps=20000)

        if not self.no_save and not os.path.exists(self.simplified_filename):
            self.simplified.save(self.simplified_filename)

        self.assertTrue(is_directed_path_graph(self.simplified.graph()), msg="The simplified model is not a path.")

    def test_correctness_simplified(self):

        print(f"Simplified model number of parameters: {self.simplified.parameter_count:,}")
        errors = []
        expected = []
        actual = []
        full_inputs = []
        while len(expected) < 10:
            x, y, full_input = TestUATok2TokBuldingSteps.prep_random_input(self)

            # prevent repetitions
            if any([(x == fi[: len(x)]).all() for fi in full_inputs]):
                continue

            full_inputs.append(full_input)
            expected.append(y)

        num_processes = min(multiprocessing.cpu_count() // 2, len(full_inputs))
        pool = multiprocessing.Pool(processes=num_processes)
        full_inputs_split = np.array_split(full_inputs, num_processes)
        results = pool.starmap(
            run_instance, [(self.simplified_filename, inpts, self.seq_len) for inpts in full_inputs_split]
        )
        pool.close()
        pool.join()
        results = np.array(results).squeeze()

        for r in results:
            actual.append(r)
            errors.append((r != y).sum())

        for exp, act in zip(expected, results):
            print(f"expected:{tuple(exp)} actual:{tuple(act)}")

        self.assertTrue(np.sum((np.array(results).squeeze() - expected) != 0) == 0)


def run_instance(filename, full_inputs, seq_len):

    loop = lsrl.ForEach.load(filename)
    results = []
    for full_input in full_inputs:
        output = loop(full_input[None, :])
        results.append(output.numpy()[:, -seq_len:].flatten())
    return results


if __name__ == "__main__":
    unittest.main()
