/*CUDA ERROR HANDLER*/
#ifndef ERRHANDLER_CUH
#define ERRHANDLER_CUH

#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuErrchk2(ans) {if (gpuAssert((ans), __FILE__, __LINE__)) return;}

/*CUDA ERROR HANDLER*/
inline int gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      return code;
   }
   return 0;
}

#define MB_SIZE (1<<20)

#endif