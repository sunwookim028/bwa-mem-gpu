#define CUDA_BLOCKSIZE 32
#define MAX_SEQLEN 8

#include "../bwa.h"
#include "../bwt.h"
#include "../bntseq.h"
#include "../bwamem.h"

/* list of pointers to data on GPU */
typedef struct {
	// constant pointers
	mem_opt_t* d_opt;		// user-defined options
	bwt_t* d_bwt;			// bwt
	bntseq_t* d_bns;		
	uint8_t* d_pac; 
	void* d_buffer_pools;	// buffer pools
	mem_pestat_t* d_pes; 	// paired-end stats
	mem_pestat_t* h_pes0;	// pes0 on host for paired-end stats
	// pointers that will change each batch
	int n_seqs;				// number of reads
	bseq1_t *d_seqs;		// reads
	int* d_hash_map;		// hash map for reordering seqs
} gpu_ptrs_t;

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

