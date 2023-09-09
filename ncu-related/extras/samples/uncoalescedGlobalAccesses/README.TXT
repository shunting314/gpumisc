== Sample CUDA application for uncoalesced global memory accesses ==
Adds a floating point constant to an input array of double3 of N elements in global memory and generates an output array of double3 in global memory. 

Defines two versions of CUDA kernel
addConstDouble3 : naive version which results in uncoalesced global memory accesses 
addConstDouble  : version which treats the double3 array as a double array and avoids uncoalesced global memory accesses

Compiling the code:
==================
  > nvcc -lineinfo -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 uncoalescedGlobalAccesses.cu -o uncoalescedGlobalAccesses

Command line arguments (both are optional):
==========================================
1) <version of kernel to use> Integer value, If not specified uses 0.
          0: Use naive version of kernel addConstDouble3()
          1: Use addConstDouble() kernel
2) <N - number of elements in input array> Should be a positive number. Default value: 1048576 (1024 x 1024)

Sample usage:
============
- Run with default arguments - addConstDouble3() kernel and default value of N
  > uncoalescedGlobalAccesses

- Run with the addConstDouble() kernel and default value of N
  > uncoalescedGlobalAccesses 1

 - Run with the addConstDouble3() kernel and N=512
  > uncoalescedGlobalAccesses 0 512


Profiling the sample using Nsight Compute command line
======================================================
- Profile addConstDouble3() - the  initial version of kernel
  > ncu --set full --import-source on  -o addConstDouble3.ncu-rep ./uncoalescedGlobalAccesses

- Profile addConstDouble() - the  updated version of the kernel
  > ncu --set full --import-source on  -o addConstDouble.ncu-rep ./uncoalescedGlobalAccesses 1

The profiler report files for the sample are also provided and they can be opened in the 
Nsight Compute UI using the "File->Open" menu option.
