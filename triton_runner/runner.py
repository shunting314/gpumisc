from triton._C.libtriton.triton import ir

ttir_path = "/data/home/shunting/gpumisc/triton_runner/sin.ttir"
context = ir.context()
ttir_module = ir.parse_mlir_module(ttir_path, context)

print(f"Loaded ttir module ({type(ttir_module)}):\n{ttir_module}")

print("bye!")
