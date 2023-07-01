import unittest

from triton_runner.runner import run_ttir_kernel
from os import path
import torch

class TestRunner(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
