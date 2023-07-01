# TODO: make this a generic runner rather than specialize to sin_kernel
from triton._C.libtriton.triton import ir
import importlib.util
from os import path
from sys import exit
import torch
from triton.compiler import compiler
from triton.runtime.driver import driver

x = torch.arange(10).to(dtype=torch.float32).cuda()
y = torch.empty(10).to(dtype=torch.float32).cuda()
artifact_dir = path.dirname(__file__)
ttir_path = f"{artifact_dir}/sin.ttir"
so_path = f"{artifact_dir}/sin_launcher.so"
num_warps = 4
num_stages = 3
arch = 80
extern_libs = {}
kernel_name = "sin_kernel_0d1d2d"

spec = importlib.util.spec_from_file_location("__triton_launcher", so_path)
mod = importlib.util.module_from_spec(spec)
c_wrapper = mod.launch

context = ir.context()
ttir_module = ir.parse_mlir_module(ttir_path, context)
ttir_module.context = context
ttir_module = compiler.optimize_ttir(ttir_module, arch)
ttgir_module = compiler.ttir_to_ttgir(ttir_module, num_warps)
ttgir_module = compiler.optimize_ttgir(ttgir_module, num_stages, arch)
llir = compiler.ttgir_to_llir(ttgir_module, extern_libs, arch)

# note: ttgir_to_llir will change ttgir_module.
# calling get_shared_memory_size on ttgir_module before the call of ttgir_to_llir
# will result in segfault.
shared = compiler.get_shared_memory_size(ttgir_module)
print(f"#shared {shared}")
ptx = compiler.llir_to_ptx(llir, arch)
cubin = compiler.ptx_to_cubin(ptx, arch)

def wrapper_args():
    compiled_kernel = None
    stream = compiler.get_cuda_stream()

    mod, func, n_regs, n_spills = driver.utils.load_binary(kernel_name, cubin, shared, 0)
    print(f"n_regs {n_regs}, n_spills {n_spills}")
    return (
        1, # grid[0]
        1, # grid[1]
        1, # grid[2]
        num_warps,
        shared,
        stream,
        func,
        None, # enter_hook
        None, # exit_hook
        compiled_kernel,
        x,
        y,
        x.numel(),
    )

c_wrapper(*wrapper_args())
print(f"y is {y}")
print(f"sin(x) is {x.sin()}")
print(f"cos(x) is {x.cos()}")
