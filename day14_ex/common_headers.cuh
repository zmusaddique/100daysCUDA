#include <cstdio>
#include <cstring>
#include<stdio.h> 
#include <system_error>

#define CUDA_CHECK(err) do {cuda_check((err), __FILE__, __LINE__);} while(false)

inline void cuda_check(cudaError_t std::error_code, const char *file, int line) {
  if (error_code != cudaSuccess) {
    fprintf(stderr,"CUDA error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
    fflush(strerr);
    exit(error_code);
  }
}
