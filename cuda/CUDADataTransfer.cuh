#define SEQ_MAX_COUNT 100000		// max number of seqs 
#define SEQ_NAME_LIMIT 5000000 		// chunk size of name
#define SEQ_COMMENT_LIMIT 100000000	// chunk size of comment
#define SEQ_LIMIT 100000000			// chunk size of seq
#define SEQ_QUAL_LIMIT 100000000	// chunk size of qual
#define SEQ_SAM_LIMIT 500000000		// chunk size of sam output

#include "../bwa.h"
#include "../bwt.h"
#include "../bntseq.h"
#include "../bwamem.h"

extern bseq1_t *preallocated_seqs;	// a chunk of SEQ_MAX_COUNT seqs on host's pinned memory
extern bseq1_t *d_preallocated_seqs;// same chunk on device
extern char *seq_name_ptr, *seq_comment_ptr, *seq_ptr, *seq_qual_ptr, *seq_sam_ptr;		 // pointers to chunks on host
extern char *d_seq_name_ptr, *d_seq_comment_ptr, *d_seq_ptr, *d_seq_qual_ptr;// pointers to chunks on device
extern int seq_name_offset, seq_comment_offset, seq_offset, seq_qual_offset; // offset on the chunks

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
} gpu_ptrs_t;

#ifdef __cplusplus
extern "C"{
#endif
	void CUDAInitSeqsMemory();

	void CUDATransferStaticData(
		const mem_opt_t *opt, 
		const bwt_t *bwt, 
		const bntseq_t *bns, 
		const uint8_t *pac,
		mem_pestat_t *pes0,
		gpu_ptrs_t *gpu_data);

	void CUDATransferSeqs(int n_seqs);

	void CUDADataFree();
#ifdef __cplusplus
}
#endif
