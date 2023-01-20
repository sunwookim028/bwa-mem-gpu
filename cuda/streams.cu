#include "streams.cuh"
#include "errHandler.cuh"
#include "CUDAKernel_memmgnt.cuh"
#include "batch_config.h"


/* transfer index data */
static void transferIndex(
	const bwt_t *bwt, 
	const bntseq_t *bns, 
	const uint8_t *pac,
	const kmers_bucket_t *kmerHashTab,
	process_data_t *process_instance)
{
		/* CUDA GLOBAL MEMORY ALLOCATION AND TRANSFER */
	unsigned long long total_size = bwt->bwt_size*sizeof(uint32_t) + bwt->n_sa*sizeof(bwtint_t) + bns->n_seqs*sizeof(bntann1_t) + bns->n_holes*sizeof(bntamb1_t) + bns->l_pac*sizeof(uint8_t);
	fprintf(stderr, "[M::%-25s] Device memory for Index ...... %.2f MB \n", __func__, (float)total_size/MB_SIZE);

	// Burrows-Wheeler Transform
		// 1. bwt_t structure
	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** bwt .......... %.2f MB\n", __func__, (float)sizeof(bwt_t)/MB_SIZE);
	bwt_t* d_bwt;
	cudaMalloc((void**)&d_bwt, sizeof(bwt_t));
	cudaMemcpy(d_bwt, bwt, sizeof(bwt_t), cudaMemcpyHostToDevice);
		// 2. int array of bwt
	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** bwt_int ...... %.2f MB\n", __func__, (float)bwt->bwt_size*sizeof(uint32_t)/MB_SIZE);
	uint32_t* d_bwt_int ;
	cudaMalloc((void**)&d_bwt_int, bwt->bwt_size*sizeof(uint32_t));
	cudaMemcpy(d_bwt_int, bwt->bwt, bwt->bwt_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
		// 3. int array of Suffix Array
	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** suffix array . %.2f MB \n", __func__, (float)bwt->n_sa*sizeof(bwtint_t)/MB_SIZE);
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
	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** bns.anns ..... %.2f MB\n", __func__, (float)bns->n_seqs*sizeof(bntann1_t)/MB_SIZE);
	cudaMalloc((void**)&temp_d_anns, bns->n_seqs*sizeof(bntann1_t));
	cudaMemcpy(temp_d_anns, h_bns->anns, bns->n_seqs*sizeof(bntann1_t), cudaMemcpyHostToDevice);
		// now assign this pointer to h_bns->anns
	h_bns->anns = temp_d_anns;

		// allocate bns->ambs on device and copy data to device
	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** bns.ambs ..... %.2f MB\n", __func__, (float)bns->n_holes*sizeof(bntamb1_t)/MB_SIZE);
	cudaMalloc((void**)&h_bns->ambs, bns->n_holes*sizeof(bntamb1_t));
	cudaMemcpy(h_bns->ambs, bns->ambs, bns->n_holes*sizeof(bntamb1_t), cudaMemcpyHostToDevice);

		// finally allocate d_bns and copy from h_bns
	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** bns .......... %.2f MB\n", __func__, (float)sizeof(bntseq_t)/MB_SIZE);
	bntseq_t* d_bns;
	cudaMalloc((void**)&d_bns, sizeof(bntseq_t));
	cudaMemcpy(d_bns, h_bns, sizeof(bntseq_t), cudaMemcpyHostToDevice);

	// PAC
	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** pac .......... %.2f MB\n", __func__, (float)bns->l_pac*sizeof(uint8_t)/MB_SIZE);
	uint8_t* d_pac ;
	cudaMalloc((void**)&d_pac, bns->l_pac/4*sizeof(uint8_t)); 		// l_pac is length of ref seq
	cudaMemcpy(d_pac, pac, bns->l_pac/4*sizeof(uint8_t), cudaMemcpyHostToDevice); 		// divide by 4 because 2-bit encoding

	// K-MER HASH TABLE
	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** kmer ......... %.2f MB\n", __func__, (float)pow4(KMER_K)*sizeof(kmers_bucket_t)/MB_SIZE);
	kmers_bucket_t* d_kmerHashTab ;
	cudaMalloc((void**)&d_kmerHashTab, pow4(KMER_K)*sizeof(kmers_bucket_t)); 		// l_pac is length of ref seq
	cudaMemcpy(d_kmerHashTab, kmerHashTab, pow4(KMER_K)*sizeof(kmers_bucket_t), cudaMemcpyHostToDevice); 		// divide by 4 because 2-bit encoding


	// output
	process_instance->d_bwt = d_bwt;
	process_instance->d_bns = d_bns;
	process_instance->d_pac = d_pac;
	process_instance->d_kmerHashTab = d_kmerHashTab;
}

/* transfer user-defined optinos */
static void transferOptions(
	const mem_opt_t *opt, 
	mem_pestat_t *pes0,
	process_data_t *process_instance)
{
	// matching and mapping options (opt)
	mem_opt_t* d_opt;
	cudaMalloc((void**)&d_opt, sizeof(mem_opt_t));
	cudaMemcpy(d_opt, opt, sizeof(mem_opt_t), cudaMemcpyHostToDevice);

	// paired-end stats: only allocate on device
	mem_pestat_t* d_pes;
	if (opt->flag&MEM_F_PE){
		fprintf(stderr, "[M::%-25s] pestat ....... %.2f MB\n", __func__, (float)4*sizeof(mem_pestat_t)/MB_SIZE);
		cudaMalloc((void**)&d_pes, 4*sizeof(mem_pestat_t));
	}

	// output
	process_instance->d_opt = d_opt;
	process_instance->d_pes = d_pes;
	process_instance->h_pes0 = pes0;
}


/* allocate memory for intermediate data on GPU
	send pointer to process_instance
 */
void allocateIntermediateData(process_data_t *process_instance){
	unsigned long long total_size = MB_MAX_COUNT*sizeof(smem_aux_t) + MB_MAX_COUNT*sizeof(mem_seed_v) + MB_MAX_COUNT*sizeof(mem_chain_v) + MB_MAX_COUNT*500*sizeof(seed_record_t) + MB_MAX_COUNT*sizeof(mem_alnreg_v) + MB_MAX_COUNT*sizeof(mem_aln_v) + 4*5*MB_MAX_COUNT*sizeof(int);
	fprintf(stderr, "[M::%-25s] total intermediate data ..... %.2f MB\n", __func__, (float)total_size/MB_SIZE);
	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** aux intervals ..... %ld MB\n", __func__, MB_MAX_COUNT*sizeof(smem_aux_t)/MB_SIZE);
	gpuErrchk( cudaMalloc((void**)&(process_instance->d_aux), MB_MAX_COUNT*sizeof(smem_aux_t)) );

	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** seeds array  ...... %ld MB\n", __func__, MB_MAX_COUNT*sizeof(mem_seed_v)/MB_SIZE);
	gpuErrchk( cudaMalloc((void**)&(process_instance->d_seq_seeds), MB_MAX_COUNT*sizeof(mem_seed_v)) );

	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** chains ............ %ld MB\n", __func__, MB_MAX_COUNT*sizeof(mem_chain_v)/MB_SIZE);
	gpuErrchk( cudaMalloc((void**)&(process_instance->d_chains), MB_MAX_COUNT*sizeof(mem_chain_v)) );

	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** seed records ...... %ld MB\n", __func__, MB_MAX_COUNT*500*sizeof(seed_record_t)/MB_SIZE);
	gpuErrchk( cudaMalloc((void**)&(process_instance->d_seed_records), MB_MAX_COUNT*500*sizeof(seed_record_t)) );	// allocate enough for all seeds

	gpuErrchk( cudaMalloc((void**)&(process_instance->d_Nseeds), sizeof(int)) );

	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** alignment regs .... %ld MB\n", __func__, MB_MAX_COUNT*sizeof(mem_alnreg_v)/MB_SIZE);
	gpuErrchk( cudaMalloc((void**)&(process_instance->d_regs), MB_MAX_COUNT*sizeof(mem_alnreg_v)) );

	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** alignments ...... %ld MB\n", __func__, MB_MAX_COUNT*sizeof(mem_aln_v)/MB_SIZE);
	gpuErrchk( cudaMalloc((void**)&(process_instance->d_alns), MB_MAX_COUNT*sizeof(mem_aln_v)) );

	if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** sorting keys .... %ld MB\n", __func__, 4*5*MB_MAX_COUNT*sizeof(int)/MB_SIZE);
	gpuErrchk( cudaMalloc((void**)&process_instance->d_sortkeys_in, MB_MAX_COUNT*5*sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&process_instance->d_sortkeys_out, MB_MAX_COUNT*5*sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&process_instance->d_seqIDs_in, MB_MAX_COUNT*5*sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&process_instance->d_seqIDs_out, MB_MAX_COUNT*5*sizeof(int)) );
}

process_data_t* newProcess(
	const mem_opt_t *opt, 
	mem_pestat_t *pes0,
	const bwt_t *bwt, 
	const bntseq_t *bns, 
	const uint8_t *pac,
	const kmers_bucket_t *kmerHashTab
)
{
    // new instance in memory
    process_data_t *instance = (process_data_t*)calloc(1, sizeof(process_data_t));

	// user-defined options
	transferOptions(opt, pes0, instance);
    
	// transfer index data
	transferIndex(bwt, bns, pac, kmerHashTab, instance);

	// init memory management
	instance->d_buffer_pools = CUDA_BufferInit();

	// initialize intermediate processing memory on device
	allocateIntermediateData(instance);

	// initialize pinned memory for reads on host
	gpuErrchk( cudaMallocHost((void**)&instance->h_seqs, MB_MAX_COUNT*sizeof(bseq1_t)) );
	gpuErrchk( cudaMallocHost((void**)&instance->h_seq_name_ptr, MB_NAME_LIMIT) );
	gpuErrchk( cudaMallocHost((void**)&instance->h_seq_comment_ptr, MB_COMMENT_LIMIT) );
	gpuErrchk( cudaMallocHost((void**)&instance->h_seq_seq_ptr, MB_SEQ_LIMIT) );
	gpuErrchk( cudaMallocHost((void**)&instance->h_seq_qual_ptr, MB_QUAL_LIMIT) );
	gpuErrchk( cudaMallocHost((void**)&instance->h_seq_sam_ptr, MB_SAM_LIMIT) );

	if (instance->h_seqs == nullptr || instance->h_seq_name_ptr == nullptr ||
		instance->h_seq_comment_ptr == nullptr || instance->h_seq_seq_ptr == nullptr ||
		instance->h_seq_qual_ptr == nullptr || instance->h_seq_sam_ptr == nullptr)
	{
        fprintf(stderr, "[M::%-25s] can't malloc minibatch on host\n", __func__);
        exit(1);
    }

	// initialize memory for reads on device
	unsigned long long total_size = MB_MAX_COUNT*sizeof(bseq1_t) + MB_NAME_LIMIT + MB_COMMENT_LIMIT + MB_SEQ_LIMIT + MB_QUAL_LIMIT + MB_SAM_LIMIT;
	fprintf(stderr, "[M::%-25s] d_seqs (process) . %llu MB\n", __func__, total_size/MB_SIZE);
	gpuErrchk( cudaMalloc((void**)&instance->d_seqs, MB_MAX_COUNT*sizeof(bseq1_t)) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_name_ptr, MB_NAME_LIMIT) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_comment_ptr, MB_COMMENT_LIMIT) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_seq_ptr, MB_SEQ_LIMIT) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_qual_ptr, MB_QUAL_LIMIT) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_sam_ptr, MB_SAM_LIMIT) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_sam_size, sizeof(int)) );

	if (instance->d_seqs == nullptr || instance->d_seq_name_ptr == nullptr ||
		instance->d_seq_comment_ptr == nullptr || instance->d_seq_seq_ptr == nullptr ||
		instance->d_seq_qual_ptr == nullptr || instance->d_seq_sam_ptr == nullptr)
	{
        fprintf(stderr, "[M::%-25s] can't malloc minibatch on GPU\n", __func__);
        exit(1);
    }

	// initialize a cuda stream for processing
	instance->CUDA_stream = malloc(sizeof(cudaStream_t));
	cudaStreamCreate((cudaStream_t*)instance->CUDA_stream);

    return instance;
}


transfer_data_t* newTransfer(){
    transfer_data_t *instance = (transfer_data_t*)calloc(1, sizeof(transfer_data_t));

	// initialize pinned memory for reads on host
	gpuErrchk( cudaMallocHost((void**)&instance->h_seqs, MB_MAX_COUNT*sizeof(bseq1_t)) );
	gpuErrchk( cudaMallocHost((void**)&instance->h_seq_name_ptr, MB_NAME_LIMIT) );
	gpuErrchk( cudaMallocHost((void**)&instance->h_seq_comment_ptr, MB_COMMENT_LIMIT) );
	gpuErrchk( cudaMallocHost((void**)&instance->h_seq_seq_ptr, MB_SEQ_LIMIT) );
	gpuErrchk( cudaMallocHost((void**)&instance->h_seq_qual_ptr, MB_QUAL_LIMIT) );
	gpuErrchk( cudaMallocHost((void**)&instance->h_seq_sam_ptr, MB_SAM_LIMIT) );
	
	if (instance->h_seqs == nullptr || instance->h_seq_name_ptr == nullptr ||
		instance->h_seq_comment_ptr == nullptr || instance->h_seq_seq_ptr == nullptr ||
		instance->h_seq_qual_ptr == nullptr || instance->h_seq_sam_ptr == nullptr)
	{
        fprintf(stderr, "[M::%-25s] can't malloc minibatch on host\n", __func__);
        exit(1);
    }

	// initialize memory for reads on device
	unsigned long long total_size = MB_MAX_COUNT*sizeof(bseq1_t) + MB_NAME_LIMIT + MB_COMMENT_LIMIT + MB_SEQ_LIMIT + MB_QUAL_LIMIT + MB_SAM_LIMIT;
	fprintf(stderr, "[M::%-25s] d_seqs (transf) .. %llu MB\n", __func__, total_size/MB_SIZE);
	gpuErrchk( cudaMalloc((void**)&instance->d_seqs, MB_MAX_COUNT*sizeof(bseq1_t)) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_name_ptr, MB_NAME_LIMIT) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_comment_ptr, MB_COMMENT_LIMIT) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_seq_ptr, MB_SEQ_LIMIT) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_qual_ptr, MB_QUAL_LIMIT) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_sam_ptr, MB_SAM_LIMIT) );
	gpuErrchk( cudaMalloc((void**)&instance->d_seq_sam_size, sizeof(int)) );

	if (instance->d_seqs == nullptr || instance->d_seq_name_ptr == nullptr ||
		instance->d_seq_comment_ptr == nullptr || instance->d_seq_seq_ptr == nullptr ||
		instance->d_seq_qual_ptr == nullptr || instance->d_seq_sam_ptr == nullptr)
	{
        fprintf(stderr, "[M::%-25s] can't malloc minibatch on GPU\n", __func__);
        exit(1);
    }

	// initialize a cuda stream for transfer
	instance->CUDA_stream = malloc(sizeof(cudaStream_t));
	cudaStreamCreate((cudaStream_t*)instance->CUDA_stream);

    return instance;
}



void swapData(process_data_t *process_data, transfer_data_t *transfer_data){
	// swap host pointers
	{ auto tmp = process_data->h_seqs; process_data->h_seqs = transfer_data->h_seqs; transfer_data->h_seqs = tmp; }
	{ auto tmp = process_data->h_seq_name_ptr; process_data->h_seq_name_ptr = transfer_data->h_seq_name_ptr; transfer_data->h_seq_name_ptr = tmp; }
	{ auto tmp = process_data->h_seq_comment_ptr; process_data->h_seq_comment_ptr = transfer_data->h_seq_comment_ptr; transfer_data->h_seq_comment_ptr = tmp; }
	{ auto tmp = process_data->h_seq_seq_ptr; process_data->h_seq_seq_ptr = transfer_data->h_seq_seq_ptr; transfer_data->h_seq_seq_ptr = tmp; }
	{ auto tmp = process_data->h_seq_qual_ptr; process_data->h_seq_qual_ptr = transfer_data->h_seq_qual_ptr; transfer_data->h_seq_qual_ptr = tmp; }
	{ auto tmp = process_data->h_seq_sam_ptr; process_data->h_seq_sam_ptr = transfer_data->h_seq_sam_ptr; transfer_data->h_seq_sam_ptr = tmp; }
	// swap device pointers
	{ auto tmp = process_data->d_seqs; process_data->d_seqs = transfer_data->d_seqs; transfer_data->d_seqs = tmp; }
	{ auto tmp = process_data->d_seq_name_ptr; process_data->d_seq_name_ptr = transfer_data->d_seq_name_ptr; transfer_data->d_seq_name_ptr = tmp; }
	{ auto tmp = process_data->d_seq_comment_ptr; process_data->d_seq_comment_ptr = transfer_data->d_seq_comment_ptr; transfer_data->d_seq_comment_ptr = tmp; }
	{ auto tmp = process_data->d_seq_seq_ptr; process_data->d_seq_seq_ptr = transfer_data->d_seq_seq_ptr; transfer_data->d_seq_seq_ptr = tmp; }
	{ auto tmp = process_data->d_seq_qual_ptr; process_data->d_seq_qual_ptr = transfer_data->d_seq_qual_ptr; transfer_data->d_seq_qual_ptr = tmp; }
	{ auto tmp = process_data->d_seq_sam_ptr; process_data->d_seq_sam_ptr = transfer_data->d_seq_sam_ptr; transfer_data->d_seq_sam_ptr = tmp; }
		// swap pointer to sam_size
	{ auto tmp = process_data->d_seq_sam_size; process_data->d_seq_sam_size = transfer_data->d_seq_sam_size; transfer_data->d_seq_sam_size = tmp; }
	// swap n_seqs
	{ auto tmp = process_data->n_seqs; process_data->n_seqs = transfer_data->n_seqs; transfer_data->n_seqs = tmp; }
    return;
}

void CUDATransferSeqsIn(transfer_data_t *transfer_data){
	cudaStream_t *transfer_stream = (cudaStream_t*)(transfer_data->CUDA_stream);
	// copy seqs to device
	gpuErrchk( cudaMemcpyAsync(transfer_data->d_seqs, transfer_data->h_seqs, transfer_data->n_seqs*sizeof(bseq1_t), cudaMemcpyHostToDevice, *transfer_stream) );
	// copy name, seq, comment, qual to device
	gpuErrchk( cudaMemcpyAsync(transfer_data->d_seq_name_ptr, transfer_data->h_seq_name_ptr, transfer_data->h_seq_name_size, cudaMemcpyHostToDevice, *transfer_stream) );
	gpuErrchk( cudaMemcpyAsync(transfer_data->d_seq_seq_ptr, transfer_data->h_seq_seq_ptr, transfer_data->h_seq_seq_size, cudaMemcpyHostToDevice, *transfer_stream) );
	gpuErrchk( cudaMemcpyAsync(transfer_data->d_seq_comment_ptr, transfer_data->h_seq_comment_ptr, transfer_data->h_seq_comment_size, cudaMemcpyHostToDevice, *transfer_stream) );
	gpuErrchk( cudaMemcpyAsync(transfer_data->d_seq_qual_ptr, transfer_data->h_seq_qual_ptr, transfer_data->h_seq_qual_size, cudaMemcpyHostToDevice, *transfer_stream) );

	gpuErrchk( cudaStreamSynchronize(*transfer_stream) );
}

/* copy sam output to host */
void CUDATransferSamOut(transfer_data_t *transfer_data){
	cudaStream_t *transfer_stream = (cudaStream_t*)(transfer_data->CUDA_stream);
	gpuErrchk( cudaMemcpyAsync(transfer_data->h_seqs, transfer_data->d_seqs, transfer_data->n_seqs*sizeof(bseq1_t), cudaMemcpyDeviceToHost, *transfer_stream) );
	// transfer all SAM from device to host
	// first find the total size of all SAM's
	int sam_size;
	gpuErrchk( cudaMemcpyAsync(&sam_size, transfer_data->d_seq_sam_size, sizeof(int), cudaMemcpyDeviceToHost, *transfer_stream) );
	// now copy
	gpuErrchk( cudaMemcpyAsync(transfer_data->h_seq_sam_ptr, transfer_data->d_seq_sam_ptr, sam_size, cudaMemcpyDeviceToHost, *transfer_stream) );

	cudaStreamSynchronize(*transfer_stream);

	// after GPU processing, seqs[i].sam are offset. Now we need to convert this offset to actual location
	bseq1_t *seqs = transfer_data->h_seqs;
	char *sam = transfer_data->h_seq_sam_ptr;
	for (int i=0; i<transfer_data->n_seqs; i++)
		seqs[i].sam = sam + (long)seqs[i].sam;
}



void resetProcess(process_data_t *process_data){
	cudaStream_t *process_stream = (cudaStream_t*)(process_data->CUDA_stream);
	// reset memory management
	CUDAResetBufferPool(process_data->d_buffer_pools, *process_stream);

	// reset intermediate data
		// reset seeds
	gpuErrchk( cudaMemsetAsync(process_data->d_Nseeds, 0, sizeof(int), *process_stream) );
		// reset sam size on device
	gpuErrchk( cudaMemsetAsync(process_data->d_seq_sam_size, 0, sizeof(int), *process_stream) );

}

void resetTransfer(transfer_data_t *transfer_data){
	// reset name, seq, comment, qual sizes
	transfer_data->h_seq_name_size = 0;
	transfer_data->h_seq_seq_size = 0;
	transfer_data->h_seq_comment_size = 0;
	transfer_data->h_seq_qual_size = 0;
	// reset n_seqs
	transfer_data->n_seqs = 0;
}