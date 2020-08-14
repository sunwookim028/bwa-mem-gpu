#define NBUFFERPOOLS 32 	// number of buffer pools
#define POOLSIZE 314572800	// size of each buffer pool is 300MB
#include <stdio.h>
#include <stdint.h>

typedef struct
{
	int lock;					// malloc lock, 0 is free
	unsigned current_offset;	// current offset to the available part of the chunk
	unsigned end_offset;		// the max offset of the chunk
} CUDAKernel_mem_info;

// initialize 32 chunks of memory
extern __host__ void* CUDA_mem_init();
// reset buffer pools
extern __host__ void CUDAResetBufferPool(void* big_pool);

/* FUNCTION TO DO MALLOC AND REALLOC WITHIN CUDA KERNELS */
// select a buffer pool from the big pool
extern __device__ void* CUDAKernelSelectPool(void* big_pool, int i);
// malloc within kernel
extern __device__ void* CUDAKernelMalloc(void* d_mem_chunk_ptr, size_t size, uint8_t align_size);
extern __device__ void* CUDAKernelCalloc(void* d_mem_chunk_ptr, size_t num, size_t size, uint8_t align_size);
// realloc within kernel
extern __device__ void* CUDAKernelRealloc(void* d_mem_chunk_ptr, void* d_current_ptr, size_t new_size, uint8_t align_size);
// memcpy within kernel
extern __device__ void cudaKernelMemcpy(void* from, void* to, size_t len);
// memmove within kernel
extern __device__ void cudaKernelMemmove(void* from, void* to, size_t len);
// check size of a chunk starting with ptr
extern __device__ unsigned cudaKernelSizeOf(void* ptr);

/*CUDA ERROR HANDLER*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
/*CUDA ERROR HANDLER*/
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
