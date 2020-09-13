#define CUDA_BLOCKSIZE 32
#define MAX_SEQLEN 8

#include "../bwa.h"
#include "../bwt.h"
#include "../bntseq.h"
#include "../bwamem.h"
#include "CUDADataTransfer.cuh"

#ifdef __cplusplus
extern "C"{
#endif
	gpu_ptrs_t GPU_Init(
		const mem_opt_t *opt, 
		const bwt_t *bwt, 
		const bntseq_t *bns, 
		const uint8_t *pac,
		mem_pestat_t *pes0
	);
	void prepare_batch_GPU(gpu_ptrs_t* gpu_data, const bseq1_t* seqs, int n_seqs, const mem_opt_t *opt);
	void mem_align_GPU(gpu_ptrs_t gpu_data, bseq1_t* seqs, const mem_opt_t *opt, const bntseq_t *bns);
#ifdef __cplusplus
} // end extern "C"
#endif

