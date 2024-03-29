import unittest
import triton.language as tl
from triton_runner.runner import run_py_kernel, run_ttir_kernel, run_ttgir_kernel, run_llir_kernel, run_ptx_kernel
from collections import namedtuple
import torch
import triton
from os import path

X_BLOCK_SIZE = 2
R_BLOCK_SIZE = 256  # may not be optimal or not equal to the default picked by inductor
NUMROW = 4
NUMCOL = 2048

class TestRunner(unittest.TestCase):
    def test_py_sumrow(self):
        """
        sumrow kernel generated by inductor
        """
        def sumrow_kernel(
            in_ptr, out_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
            xnumel = 4
            rnumel = 2048
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
            xmask = xindex < xnumel
            rbase = tl.arange(0, RBLOCK)[None, :]
            x0 = xindex
            _tmp1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
            for roffset in range(0, rnumel, RBLOCK):
                rindex = roffset + rbase
                rmask = rindex < rnumel
                r1 = rindex
                tmp0 = tl.load(in_ptr + (r1 + 2048 * x0), rmask & xmask, other=0)
                tmp2 = _tmp1 + tmp0
                _tmp1 = tl.where(xmask & rmask, tmp2, _tmp1)
            tmp1 = tl.sum(_tmp1, 1)[:, None]
            tl.store(out_ptr + x0, tmp1, xmask)

        run_py_kernel(
            py_fn=sumrow_kernel,
            instance_desc=namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"])((0, 1, 3), ()),
            constants={4: X_BLOCK_SIZE, 5: R_BLOCK_SIZE},
            kernel_name="sumrow_kernel_0d1d23d",
            signature={0: "*fp32", 1: "*fp32", 2: "i32", 3: "i32"},
            grid_x=triton.cdiv(NUMROW, X_BLOCK_SIZE),
            args=lambda: (
                torch.rand(NUMROW, NUMCOL, device="cuda"),
                torch.empty(NUMROW, device="cuda"),
                NUMROW,
                NUMCOL,
            ),
            verifier=lambda x, y, xnumel, rnumel: (
                print(f"expected sum {x.sum(dim=-1).sum()}"),
                print(f"actual sum {y.sum()}"),
                self.assertTrue(torch.allclose(y, x.sum(dim=-1))),
            ),
        )

    def test_ttir_sumrow(self):
        run_ttir_kernel(
            ttir_path=f"{path.dirname(__file__)}/../triton_runner/sumrow.ttir",
            kernel_name="sumrow_kernel_0d1d23d",
            signature={0: "*fp32", 1: "*fp32", 2: "i32", 3: "i32"},
            grid_x=triton.cdiv(NUMROW, X_BLOCK_SIZE),
            args=lambda: (
                torch.rand(NUMROW, NUMCOL, device="cuda"),
                torch.empty(NUMROW, device="cuda"),
                NUMROW,
                NUMCOL,
            ),
            verifier=lambda x, y, xnumel, rnumel: (
                print(f"expected sum {x.sum(dim=-1).sum()}"),
                print(f"actual sum {y.sum()}"),
                self.assertTrue(torch.allclose(y, x.sum(dim=-1))),
            ),
        )

    def test_ttgir_sumrow(self):
        run_ttgir_kernel(
            ttgir_path=f"{path.dirname(__file__)}/../triton_runner/sumrow.ttgir",
            kernel_name="sumrow_kernel_0d1d23d",
            signature={0: "*fp32", 1: "*fp32", 2: "i32", 3: "i32"},
            grid_x=triton.cdiv(NUMROW, X_BLOCK_SIZE),
            args=lambda: (
                torch.rand(NUMROW, NUMCOL, device="cuda"),
                torch.empty(NUMROW, device="cuda"),
                NUMROW,
                NUMCOL,
            ),
            verifier=lambda x, y, xnumel, rnumel: (
                print(f"expected sum {x.sum(dim=-1).sum()}"),
                print(f"actual sum {y.sum()}"),
                self.assertTrue(torch.allclose(y, x.sum(dim=-1))),
            ),
        )

    def test_llir_sumrow(self):
        run_llir_kernel(
            llir=f"{path.dirname(__file__)}/../triton_runner/sumrow.llir",
            kernel_name="sumrow_kernel_0d1d23d",
            signature={0: "*fp32", 1: "*fp32", 2: "i32", 3: "i32"},
            grid_x=triton.cdiv(NUMROW, X_BLOCK_SIZE),
            args=lambda: (
                torch.rand(NUMROW, NUMCOL, device="cuda"),
                torch.empty(NUMROW, device="cuda"),
                NUMROW,
                NUMCOL,
            ),
            verifier=lambda x, y, xnumel, rnumel: (
                print(f"expected sum {x.sum(dim=-1).sum()}"),
                print(f"actual sum {y.sum()}"),
                self.assertTrue(torch.allclose(y, x.sum(dim=-1))),
            ),
            shared=512,
        )

    def test_ptx_sumrow(self):
        run_ptx_kernel(
            ptx=f"{path.dirname(__file__)}/../triton_runner/sumrow.ptx",
            kernel_name="sumrow_kernel_0d1d23d",
            signature={0: "*fp32", 1: "*fp32", 2: "i32", 3: "i32"},
            grid_x=triton.cdiv(NUMROW, X_BLOCK_SIZE),
            args=lambda: (
                torch.rand(NUMROW, NUMCOL, device="cuda"),
                torch.empty(NUMROW, device="cuda"),
                NUMROW,
                NUMCOL,
            ),
            verifier=lambda x, y, xnumel, rnumel: (
                print(f"expected sum {x.sum(dim=-1).sum()}"),
                print(f"actual sum {y.sum()}"),
                self.assertTrue(torch.allclose(y, x.sum(dim=-1))),
            ),
            shared=512,
        )


if __name__ == "__main__":
    unittest.main()
