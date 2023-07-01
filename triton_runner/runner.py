from triton._C.libtriton.triton import ir
import importlib.util
from sys import exit
import torch
from triton.compiler import compiler, make_launcher
from triton.runtime.driver import driver
import functools

num_warps = 4
num_stages = 3
extern_libs = {}

# without this we get "RuntimeError: Triton Error [CUDA]: invalid device context"
# when calling driver.utils.load_binary
torch.empty(1).cuda()

@functools.lru_cache(None)
def get_arch():
    dev_id = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev_id)
    return prop.major * 10 + prop.minor

# TODO avoid redundancy between arguments
def run_ttir_kernel(
    ttir_path,
    kernel_name,
    signature,
    grid_x,
    args,
    verifier,
):
    arch = get_arch()
    device_id = torch.cuda.current_device()
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
    ptx = compiler.llir_to_ptx(llir, arch)
    cubin = compiler.ptx_to_cubin(ptx, arch)

    # load
    cu_mod, cu_func, n_regs, n_spills = driver.utils.load_binary(kernel_name, cubin, shared, device_id)

    # launcher so
    launcher_so = make_launcher.make_stub(kernel_name, signature, constants={});
    spec = importlib.util.spec_from_file_location("__triton_launcher", launcher_so)
    launcher_mod = importlib.util.module_from_spec(spec)
    launcher_c_wrapper = launcher_mod.launch

    stream = compiler.get_cuda_stream()
    args = args() if callable(args) else args
    launcher_c_wrapper(
        grid_x,
        1,
        1,
        num_warps,
        shared,
        stream,
        cu_func,
        None, # enter_hook
        None, # exit_hook
        None, # compiled_kernel
        *args,
    )
    verifier(*args)
