#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  cudaDeviceProp prop;
  cudaError_t status = cudaGetDeviceProperties(&prop, 0);

  if (status != cudaSuccess) {
    printf("%s", cudaGetErrorString(status));
    return 1;
  }

  int v = prop.major * 10 + prop.minor;
  printf("-gencode arch=compute_%d,code=sm_%d\n", v, v);

  return 0;
}
