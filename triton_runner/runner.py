from triton._C.libtriton.triton import ir
import importlib.util
from sys import exit
import torch
from triton.compiler import compiler, make_launcher
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction
from triton.compiler.code_generator import ast_to_ttir
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

def run_py_kernel(
    py_fn,  # don't need the torch.jit wrapper
    instance_desc,
    signature,
    constants,
    **kwargs,
):
    jit_fn = JITFunction(py_fn)
    ttir = ast_to_ttir(jit_fn, signature, instance_desc, constants, debug=True)
    run_ttir_kernel(ttir, signature=signature, **kwargs)

def run_ttir_kernel(
    ttir_path,
    **kwargs,
):
    arch = get_arch()
    if isinstance(ttir_path, str):
        context = ir.context()
        ttir_module = ir.parse_mlir_module(ttir_path, context)
        ttir_module.context = context
    else:
        ttir_module = ttir_path  # already a ttir module
    ttir_module = compiler.optimize_ttir(ttir_module, arch)
    ttgir_module = compiler.ttir_to_ttgir(ttir_module, num_warps)
    run_ttgir_kernel(ttgir_module, **kwargs)

def run_ttgir_kernel(
    ttgir_path,
    **kwargs,
):
    arch = get_arch()
    if isinstance(ttgir_path, str):
        context = ir.context()
        ttgir_module = ir.parse_mlir_module(ttgir_path, context)
        ttgir_module.context = context
    else:
        ttgir_module = ttgir_path
    ttgir_module = compiler.optimize_ttgir(ttgir_module, num_stages, arch)
    llir = compiler.ttgir_to_llir(ttgir_module, extern_libs, arch)

    # note: ttgir_to_llir will change ttgir_module.
    # calling get_shared_memory_size on ttgir_module before the call of ttgir_to_llir
    # will result in segfault.
    shared = compiler.get_shared_memory_size(ttgir_module)

    run_llir_kernel(llir, shared=shared, **kwargs)

def run_llir_kernel(
    llir,
    **kwargs,
):
    assert isinstance(llir, str)
    if llir.endswith(".llir"):  # is a path
        with open(llir, "r") as f:
            llir = f.read()

    arch = get_arch()
    ptx = compiler.llir_to_ptx(llir, arch)
    run_ptx_kernel(ptx, **kwargs)

def run_ptx_kernel(
    ptx,
    shared,
    kernel_name,
    signature,
    grid_x,
    args,
    verifier,
):
    assert isinstance(ptx, str)
    if ptx.endswith(".ptx"):  # is a path
        with open(ptx, "r") as f:
            ptx = f.read()

    arch = get_arch()
    cubin = compiler.ptx_to_cubin(ptx, arch)

    # load
    device_id = torch.cuda.current_device()
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
