import unittest
import triton.language as tl
import triton
import torch
from collections import namedtuple
from os import path

# TODO run_ttir_kernel
from triton_runner.runner import run_py_kernel, run_cu_kernel

NUMEL = 98432
BLOCK_SIZE = 1024

class TestRunner(unittest.TestCase):
    def test_py_add(self):
        def add_kernel(
            x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

        run_py_kernel(
            py_fn=add_kernel,
            instance_desc=namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"])((0, 1, 2, 3), ()),
            constants={4: BLOCK_SIZE},
            kernel_name="add_kernel_0d1d2d3d",
            signature={0: "*fp32", 1: "*fp32", 2: "*fp32", 3: "i32"},
            grid_x=triton.cdiv(NUMEL, BLOCK_SIZE),
            args=lambda: (
                torch.rand(NUMEL, device="cuda"),
                torch.rand(NUMEL, device="cuda"),
                torch.empty(NUMEL, device="cuda"),
                NUMEL,
            ),
            verifier=lambda x, y, out, numel: (
                print(f"expected sum {(x + y).sum()}"),
                print(f"actual sum {out.sum()}"),
                self.assertTrue(torch.allclose(out, x + y)),
            ),
        )

    def test_cu_add(self):
        run_cu_kernel(
            cu_path=f"{path.dirname(__file__)}/../triton_runner/vector_add.cu",
            kernel_name="add_kernel_0d1d2d3d",
            signature={0: "*fp32", 1: "*fp32", 2: "*fp32", 3: "i32"},
            grid_x=triton.cdiv(NUMEL, BLOCK_SIZE),
            args=lambda: (
                torch.rand(NUMEL, device="cuda"),
                torch.rand(NUMEL, device="cuda"),
                torch.empty(NUMEL, device="cuda"),
                NUMEL,
            ),
            verifier=lambda x, y, out, numel: (
                print(f"expected sum {(x + y).sum()}"),
                print(f"actual sum {out.sum()}"),
                self.assertTrue(torch.allclose(out, x + y)),
            ),
            shared=0,
        )


if __name__ == "__main__":
    unittest.main()
