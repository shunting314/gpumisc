# test the version of softmax from triton tutorial
import unittest
import torch
import triton
import triton.language as tl
from os import path
from triton_runner.runner import run_py_kernel, run_cu_kernel, run_ttir_kernel
from collections import namedtuple

NUM_ROW = 1823
NUM_COL = 781
BLOCK_SIZE = 1024

class TestRunner(unittest.TestCase):
    def test_py_softmax(self):
        def softmax_kernel(
           input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
           row_idx = tl.program_id(0)
           row_start_ptr = input_ptr + row_idx * n_cols
           col_offsets = tl.arange(0, BLOCK_SIZE)
           input_ptrs = row_start_ptr + col_offsets
           row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
           row_minus_max = row - tl.max(row, axis=0)
           numerator = tl.exp(row_minus_max)
           denominator = tl.sum(numerator, axis=0)
           softmax_output = numerator / denominator
           output_row_start_ptr = output_ptr + row_idx * n_cols
           output_ptrs = output_row_start_ptr + col_offsets
           tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

        run_py_kernel(
            py_fn=softmax_kernel,
            instance_desc=namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"])((0, 1,), ()),
            constants={3: BLOCK_SIZE},
            kernel_name="softmax_kernel_0d1d2",
            signature={0: "*fp32", 1: "*fp32", 2: "i32"},
            grid_x=NUM_ROW,
            args=lambda: (
                torch.randn(NUM_ROW, NUM_COL, device="cuda"),
                torch.empty(NUM_ROW, NUM_COL, device="cuda"),
                NUM_COL,
            ),
            verifier=lambda x, out, num_col: (
                print(f"expected sum {torch.softmax(x, axis=1).sum()}"),
                print(f"actual sum {out.sum()}"),
                self.assertTrue(torch.allclose(torch.softmax(x, axis=1), out)),
            ),
        )

    def test_ttir_softmax(self):
        run_ttir_kernel(
            ttir_path=f"{path.dirname(__file__)}/../triton_runner/softmax_tutor.ttir",
            kernel_name="softmax_kernel_0d1d2",
            signature={0: "*fp32", 1: "*fp32", 2: "i32"},
            grid_x=NUM_ROW,
            args=lambda: (
                torch.randn(NUM_ROW, NUM_COL, device="cuda"),
                torch.empty(NUM_ROW, NUM_COL, device="cuda"),
                NUM_COL,
            ),
            verifier=lambda x, out, num_col: (
                print(f"expected sum {torch.softmax(x, axis=1).sum()}"),
                print(f"actual sum {out.sum()}"),
                self.assertTrue(torch.allclose(torch.softmax(x, axis=1), out)),
            ),
        )


if __name__ == "__main__":
    unittest.main()
