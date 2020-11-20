#include "kbtree_CUDA.cuh"
#include "CUDADataTransfer.cuh"
#include <stdio.h>
#include <stdint.h>
#include "CUDAKernel_memmgnt.cuh"

char *seq_name_ptr = 0; int seq_name_offset = 0;
char *seq_comment_ptr = 0; int seq_comment_offset = 0;
char *seq_ptr = 0; int seq_offset = 0;
char *seq_qual_ptr = 0; int seq_qual_offset = 0;
char *seq_sam_ptr = 0;
__device__ char *d_seq_sam_ptr = 0;
char *d_seq_name_ptr = 0;
char *d_seq_comment_ptr = 0;
char *d_seq_ptr = 0;
char *d_seq_qual_ptr = 0;
bseq1_t *preallocated_seqs=0, *d_preallocated_seqs=0;

/* Allocate big chunks of strings for seqs and seqs members name, comment, seq, qual */
void CUDAInitSeqsMemory()
{
	// allocate big chunks of memory as pinned memory on host
	gpuErrchk(cudaMallocHost((void**)&seq_name_ptr, SEQ_NAME_LIMIT));
	gpuErrchk(cudaMallocHost((void**)&seq_comment_ptr, SEQ_COMMENT_LIMIT));
	gpuErrchk(cudaMallocHost((void**)&seq_ptr, SEQ_LIMIT));
	gpuErrchk(cudaMallocHost((void**)&seq_qual_ptr, SEQ_QUAL_LIMIT));
	gpuErrchk(cudaMallocHost((void**)&preallocated_seqs, SEQ_MAX_COUNT*sizeof(bseq1_t)));
	gpuErrchk(cudaMallocHost((void**)&seq_sam_ptr, SEQ_SAM_LIMIT));
	// allocate corresponding chunks on device
	gpuErrchk(cudaMalloc((void**)&d_seq_name_ptr, SEQ_NAME_LIMIT));
	gpuErrchk(cudaMalloc((void**)&d_seq_comment_ptr, SEQ_COMMENT_LIMIT));
	gpuErrchk(cudaMalloc((void**)&d_seq_ptr, SEQ_LIMIT));
	gpuErrchk(cudaMalloc((void**)&d_seq_qual_ptr, SEQ_QUAL_LIMIT));
	gpuErrchk(cudaMalloc((void**)&d_preallocated_seqs, SEQ_MAX_COUNT*sizeof(bseq1_t)));
	char* symbol_addr;
	gpuErrchk(cudaGetSymbolAddress((void**)&symbol_addr, d_seq_sam_ptr));
	char* d_temp;
	gpuErrchk(cudaMalloc((void**)&d_temp, SEQ_SAM_LIMIT));
	gpuErrchk(cudaMemcpy(symbol_addr, &d_temp, sizeof(char*), cudaMemcpyHostToDevice));
	fprintf(stderr, "[M::%s] seq name ......... %d MB\n", __func__, (int)SEQ_NAME_LIMIT/1000000);
	fprintf(stderr, "[M::%s] seq comment ...... %d MB\n", __func__, (int)SEQ_COMMENT_LIMIT/1000000);
	fprintf(stderr, "[M::%s] seq  ............. %d MB\n", __func__, (int)SEQ_LIMIT/1000000);
	fprintf(stderr, "[M::%s] seq qual ......... %d MB\n", __func__, (int)SEQ_QUAL_LIMIT/1000000);
	fprintf(stderr, "[M::%s] seq info ......... %d MB\n", __func__, (int)SEQ_MAX_COUNT*sizeof(bseq1_t)/1000000);
	fprintf(stderr, "[M::%s] sam .............. %d MB\n", __func__, (int)SEQ_SAM_LIMIT/1000000);
}

/* transfer one-time static data */
void CUDATransferStaticData(
	const mem_opt_t *opt, 
	const bwt_t *bwt, 
	const bntseq_t *bns, 
	const uint8_t *pac,
	mem_pestat_t *pes0,
	gpu_ptrs_t *gpu_data)
{
		/* CUDA GLOBAL MEMORY ALLOCATION AND TRANSFER */
	fprintf(stderr, "[M::%s] Device memory allocation ......\n", __func__);

	// matching and mapping options (opt)
	fprintf(stderr, "[M::%s] options ...... %.2f MB\n", __func__, (float)sizeof(mem_opt_t)/1000000);
	mem_opt_t* d_opt;
	cudaMalloc((void**)&d_opt, sizeof(mem_opt_t));
	cudaMemcpy(d_opt, opt, sizeof(mem_opt_t), cudaMemcpyHostToDevice);

	// Burrows-Wheeler Transform
		// 1. bwt_t structure
	fprintf(stderr, "[M::%s] bwt .......... %.2f MB\n", __func__, (float)sizeof(bwt_t)/1000000);
	bwt_t* d_bwt;
	cudaMalloc((void**)&d_bwt, sizeof(bwt_t));
	cudaMemcpy(d_bwt, bwt, sizeof(bwt_t), cudaMemcpyHostToDevice);
		// 2. int array of bwt
	fprintf(stderr, "[M::%s] bwt_int ...... %.2f MB\n", __func__, (float)bwt->bwt_size*sizeof(uint32_t)/1000000);
	uint32_t* d_bwt_int ;
	cudaMalloc((void**)&d_bwt_int, bwt->bwt_size*sizeof(uint32_t));
	cudaMemcpy(d_bwt_int, bwt->bwt, bwt->bwt_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
		// 3. int array of Suffix Array
	fprintf(stderr, "[M::%s] suffix array . %.2f MB \n", __func__, (float)bwt->n_sa*sizeof(bwtint_t)/1000000);
	bwtint_t* d_bwt_sa ;
	cudaMalloc((void**)&d_bwt_sa, bwt->n_sa*sizeof(bwtint_t));
	cudaMemcpy(d_bwt_sa, bwt->sa, bwt->n_sa*sizeof(bwtint_t), cudaMemcpyHostToDevice);
		// set pointers on device's memory to bwt_int and SA
	cudaMemcpy((void**)&(d_bwt->bwt), &d_bwt_int, sizeof(uint32_t*), cudaMemcpyHostToDevice);
	cudaMemcpy((void**)&(d_bwt->sa), &d_bwt_sa, sizeof(bwtint_t*), cudaMemcpyHostToDevice);

	// BNS
	// First create h_bns as a copy of bns on host
	// Then allocate its member pointers on device and copy data over
	// Then copy h_bns to d_bns
	uint32_t i, size;			// loop index and length of strings
	bntseq_t* h_bns;			// host copy to modify pointers
	h_bns = (bntseq_t*)malloc(sizeof(bntseq_t));
	memcpy(h_bns, bns, sizeof(bntseq_t));
	h_bns->anns = (bntann1_t*)malloc(bns->n_seqs*sizeof(bntann1_t));
	memcpy(h_bns->ambs, bns->ambs, bns->n_holes*sizeof(bntamb1_t));
	h_bns->ambs = (bntamb1_t*)malloc(bns->n_holes*sizeof(bntamb1_t));
	memcpy(h_bns->anns, bns->anns, bns->n_seqs*sizeof(bntann1_t));

		// allocate anns.name
	for (i=0; i<bns->n_seqs; i++){
		size = strlen(bns->anns[i].name);
		// allocate this name and copy to device
		cudaMalloc((void**)&(h_bns->anns[i].name), size+1); 			// +1 for "\0"
		cudaMemcpy(h_bns->anns[i].name, bns->anns[i].name, size+1, cudaMemcpyHostToDevice);
	}
	// allocate anns.anno
	for (i=0; i<bns->n_seqs; i++){
		size = strlen(bns->anns[i].anno);
		// allocate this name and copy to device
		cudaMalloc((void**)&(h_bns->anns[i].anno), size+1); 			// +1 for "\0"
		cudaMemcpy(h_bns->anns[i].anno, bns->anns[i].anno, size+1, cudaMemcpyHostToDevice);
	}
		// now h_bns->anns has pointers of name and anno on device
		// allocate anns on device and copy data from h_bns->anns to device
	bntann1_t* temp_d_anns;
	fprintf(stderr, "[M::%s] bns.anns ..... %.2f MB\n", __func__, (float)bns->n_seqs*sizeof(bntann1_t)/1000000);
	cudaMalloc((void**)&temp_d_anns, bns->n_seqs*sizeof(bntann1_t));
	cudaMemcpy(temp_d_anns, h_bns->anns, bns->n_seqs*sizeof(bntann1_t), cudaMemcpyHostToDevice);
		// now assign this pointer to h_bns->anns
	h_bns->anns = temp_d_anns;

		// allocate bns->ambs on device and copy data to device
	fprintf(stderr, "[M::%s] bns.ambs ..... %.2f MB\n", __func__, (float)bns->n_holes*sizeof(bntamb1_t)/1000000);
	cudaMalloc((void**)&h_bns->ambs, bns->n_holes*sizeof(bntamb1_t));
	cudaMemcpy(h_bns->ambs, bns->ambs, bns->n_holes*sizeof(bntamb1_t), cudaMemcpyHostToDevice);

		// finally allocate d_bns and copy from h_bns
	fprintf(stderr, "[M::%s] bns .......... %.2f MB\n", __func__, (float)sizeof(bntseq_t)/1000000);
	bntseq_t* d_bns;
	cudaMalloc((void**)&d_bns, sizeof(bntseq_t));
	cudaMemcpy(d_bns, h_bns, sizeof(bntseq_t), cudaMemcpyHostToDevice);

	// PAC
	fprintf(stderr, "[M::%s] pac .......... %.2f MB\n", __func__, (float)bns->l_pac*sizeof(uint8_t)/1000000);
	uint8_t* d_pac ;
	cudaMalloc((void**)&d_pac, bns->l_pac/4*sizeof(uint8_t)); 		// l_pac is length of ref seq
	cudaMemcpy(d_pac, pac, bns->l_pac/4*sizeof(uint8_t), cudaMemcpyHostToDevice); 		// divide by 4 because 2-bit encoding

	// paired-end stats: only allocate on device
	mem_pestat_t* d_pes;
	if (opt->flag&MEM_F_PE){
		fprintf(stderr, "[M::%s] pestat ....... %.2f MB\n", __func__, (float)4*sizeof(mem_pestat_t)/1000000);
		cudaMalloc((void**)&d_pes, 4*sizeof(mem_pestat_t));
	}

	// output
	gpu_data->d_opt = d_opt;
	gpu_data->d_bwt = d_bwt;
	gpu_data->d_bns = d_bns;
	gpu_data->d_pac = d_pac;
	gpu_data->d_pes = d_pes;
	gpu_data->h_pes0 = pes0;
}

/* transfer seqs */
void CUDATransferSeqs(int n_seqs)
{
	// copy name to device
	gpuErrchk(cudaMemcpy(d_seq_name_ptr, seq_name_ptr, seq_name_offset, cudaMemcpyHostToDevice));
	// copy seq to device
	gpuErrchk(cudaMemcpy(d_seq_ptr, seq_ptr, seq_offset, cudaMemcpyHostToDevice));
	// copy comment to device
	gpuErrchk(cudaMemcpy(d_seq_comment_ptr, seq_comment_ptr, seq_comment_offset, cudaMemcpyHostToDevice));
	// copy qual to device
	gpuErrchk(cudaMemcpy(d_seq_qual_ptr, seq_qual_ptr, seq_qual_offset, cudaMemcpyHostToDevice));
	// copy seqs to device
	gpuErrchk(cudaMemcpy(d_preallocated_seqs, preallocated_seqs, n_seqs*sizeof(bseq1_t), cudaMemcpyHostToDevice));
}

/* transfer SAM output */

void CUDADataFree(){
	cudaFreeHost(preallocated_seqs);
	cudaFree(d_preallocated_seqs);
	cudaFreeHost(seq_name_ptr); cudaFreeHost(seq_comment_ptr), cudaFreeHost(seq_ptr), cudaFreeHost(seq_qual_ptr); cudaFreeHost(seq_sam_ptr);
	cudaFree(d_seq_name_ptr); cudaFree(d_seq_comment_ptr); cudaFree(d_seq_ptr); cudaFree(d_seq_qual_ptr);
}