#include "../bwa.h"
#include "../bwt.h"
#include "../bntseq.h"
#include "../bwamem.h"
#include "batch_config.h"

// collections of SA intervals
typedef struct {
	bwtintv_v mem, mem1, *tmpv[2];
} smem_aux_t;

typedef struct
{
	int seqID;			// read ID
	uint16_t chainID;	// index on the chain vector of the read
	uint16_t seedID	;	// index of seed on the chain
	uint16_t regID;		// index on the (mem_alnreg_t)regs.a vector
	// below are for SW extension
	uint8_t* read_left; 	// string of read on the left of seed
	uint8_t* ref_left;		// string of reference on the left of seed
	uint8_t* read_right; 	// string of read on the right of seed
	uint8_t* ref_right;		// string of reference on the right of seed
	uint16_t readlen_left; 	// length of read on the left of seed
	uint16_t reflen_left;	// length of reference on the left of seed
	uint16_t readlen_right; // length of read on the right of seed
	uint16_t reflen_right;	// length of reference on the right of seed
} seed_record_t;

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
	// intermediate data
	seed_record_t *d_seed_records; 	// global records of seeds, a big chunk of memory
	int *d_Nseeds;			// total number of seeds
	smem_aux_t* d_aux;		// collections of SA intervals, vector of size nseqs
	mem_seed_v* d_seq_seeds;// seeds array for each read
	mem_chain_v *d_chains;	// chain vectors of size nseqs
	mem_alnreg_v *d_regs;	// alignment info vectors, size nseqs
	mem_aln_v * d_alns;		// alignment vectors, size nseqs
	// arrays for sorting, each has length = n_seqs
	int *d_sortkeys_in;
	int *d_seqIDs_in;
	int *d_sortkeys_out;
	int *d_seqIDs_out;
	int n_sortkeys;

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
