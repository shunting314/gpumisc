#include <stdio.h>

void showDeviceProperties(int devid) {
  printf("Device %d properties:\n", devid);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devid);
  printf("  name: '%s'\n", prop.name);
  printf("  #SM: %d\n", prop.multiProcessorCount);
  printf("  #global mem: %.3fGB (%.3fGiB)\n", (double) prop.totalGlobalMem / 1e9, (double) prop.totalGlobalMem / (1 << 30));
  printf("  #L2$: %.3fMB (%.3fMiB)\n", (double) prop.l2CacheSize / 1e6, (double) prop.l2CacheSize / (1 << 20));
  printf("  maxThreadsPerBlock %d\n", prop.maxThreadsPerBlock);
  printf("  clockRate %d\n", prop.clockRate); // core clock rate
  for (int i = 0; i < 3; ++i) {
    printf("  maxThreadsDim[%d] %d\n", i, prop.maxThreadsDim[i]);
  }
  for (int i = 0; i < 3; ++i) {
    printf("  maxGridSize[%d] %d\n", i, prop.maxGridSize[i]);
  }
  printf("  regsPerBlock %d\n", prop.regsPerBlock);
  printf("  regsPerMultiprocessor %d\n", prop.regsPerMultiprocessor);
  printf("  warpSize %d\n", prop.warpSize);
  printf("  sharedMemPerBlock %d\n", prop.sharedMemPerBlock);
  printf("  sharedMemPerMultiprocessor %d\n", prop.sharedMemPerMultiprocessor);
}

int main(void) {
  int ngpu = -1;
  cudaGetDeviceCount(&ngpu);
  printf("#GPU %d\n", ngpu);

  // show device properties for each available devices
  for (int i = 0; i < ngpu; ++i) {
    showDeviceProperties(i);
  }
  return 0;
}
