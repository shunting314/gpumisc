import unittest
from collections import namedtuple

from triton_runner.runner import run_py_kernel, run_ttir_kernel, run_ttgir_kernel
from os import path
import torch
import triton
import triton.language as tl

class TestRunner(unittest.TestCase):
    def test_py_sin(self):
        def sin_kernel(in_ptr, out_ptr, numel, XBLOCK: tl.constexpr):
            xindex = tl.program_id(axis=0) * XBLOCK + tl.arange(0, XBLOCK)
            xmask = xindex < numel
            t0 = tl.load(in_ptr + xindex, xmask)
            t1 = tl.sin(t0)
            tl.store(out_ptr + xindex, t1, xmask)

        run_py_kernel(
            py_fn=sin_kernel,
            instance_desc=namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"])((0, 1, 2), ()),
            constants={3: 1024},
            kernel_name="sin_kernel_0d1d2d",
            signature={0: "*fp32", 1: "*fp32", 2: "i32"},
            grid_x=1,
            args=lambda: (
                torch.arange(10).to(dtype=torch.float32).cuda(),
                torch.empty(10).to(dtype=torch.float32).cuda(),
                10,
            ),
            verifier=lambda x, y, numel: (
                print(f"x {x}"),
                print(f"y {y}"),
                print(f"expected {x.sin()}"),
                self.assertTrue(torch.allclose(y, x.sin())),
            ),
        )

    def test_ttir_sin(self):
        run_ttir_kernel(
            ttir_path=f"{path.dirname(__file__)}/../triton_runner/sin.ttir",
            kernel_name="sin_kernel_0d1d2d",
            signature={0: "*f32", 1: "*f32", 2: "i32"},
            grid_x=1,
            args=lambda: (
                torch.arange(10).to(dtype=torch.float32).cuda(),
                torch.empty(10).to(dtype=torch.float32).cuda(),
                10,
            ),
            verifier=lambda x, y, numel: (
                print(f"x {x}"),
                print(f"y {y}"),
                print(f"expected {x.sin()}"),
                self.assertTrue(torch.allclose(y, x.sin())),
            ),
        )

    def test_ttgir_sin(self):
        run_ttgir_kernel(
            ttgir_path=f"{path.dirname(__file__)}/../triton_runner/sin.ttgir",
            kernel_name="sin_kernel_0d1d2d",
            signature={0: "*f32", 1: "*f32", 2: "i32"},
            grid_x=1,
            args=lambda: (
                torch.arange(10).to(dtype=torch.float32).cuda(),
                torch.empty(10).to(dtype=torch.float32).cuda(),
                10,
            ),
            verifier=lambda x, y, numel: (
                print(f"x {x}"),
                print(f"y {y}"),
                print(f"expected {x.sin()}"),
                self.assertTrue(torch.allclose(y, x.sin())),
            ),
        )


if __name__ == "__main__":
    unittest.main()
