#include "../bwa.h"
#include "../bwamem.h"
#include "../kvec.h"
#include "../utils.h"
#include "../bntseq.h"
#include "../kseq.h"
#include <locale.h>
#include "bwamem_GPU.cuh"
#include "batch_config.h"
KSEQ_DECLARE(gzFile)

typedef struct {
	kseq_t *ks, *ks2;
	mem_opt_t *opt;
	mem_pestat_t *pes0;
	int64_t n_processed;
	int copy_comment, actual_chunk_size;
	bwaidx_t *idx;
	gpu_ptrs_t gpu_data;
} ktp_aux_t;

typedef struct {
	ktp_aux_t *aux;
	int n_seqs;
	bseq1_t *seqs;
	int n_seqs_io;	// previous/next batch of seqs for output/input
	bseq1_t *seqs_io;
} ktp_data_t;


#ifdef __cplusplus
extern "C"{
#endif

	void processHideIO(ktp_aux_t *aux);

#ifdef __cplusplus
} // end extern "C"
#endif
