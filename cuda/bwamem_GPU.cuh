#define CUDA_BLOCKSIZE 32

#include "../bwa.h"
#include "../bwt.h"
#include "../bntseq.h"
#include "../bwamem.h"
#include "streams.cuh"

#ifdef __cplusplus
extern "C"{
#endif
	/* align reads and return the size of SAM output */
	void mem_align_GPU(process_data_t *process_data);
#ifdef __cplusplus
} // end extern "C"
#endif

