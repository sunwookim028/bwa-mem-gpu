/* functions that drive the main process (hiding IO, parallel GPUs, etc.) */

#include "process.h"
#include "streams.cuh"
#include <future>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>


/**
 * @brief convert current reads addresses (on host) to their addresses on GPU
 * 
 * @param seqs 
 * @param n_seqs 
 * @param transfer_data transfer_data_t object where these reads reside
 * @param h_seq_seq_ptr pointer to the first seq on device
 */
void convert2DevAddr(bseq1_t *reads, int n_reads, transfer_data_t *transfer_data){
	for (int i = 0; i < n_reads; i ++){
		reads[i].name = reads[i].name - transfer_data->h_seq_name_ptr + transfer_data->d_seq_name_ptr;
		reads[i].seq = reads[i].seq - transfer_data->h_seq_seq_ptr + transfer_data->d_seq_seq_ptr;
		reads[i].comment = reads[i].comment - transfer_data->h_seq_comment_ptr + transfer_data->d_seq_comment_ptr;
		reads[i].qual = reads[i].qual - transfer_data->h_seq_qual_ptr + transfer_data->d_seq_qual_ptr;
	}
}


/**
 * @brief compare 2 reads a and b. 
 * @return int positive if a > b, negative if a < b, 0 if a == b
 */
int compareReads (const void * a, const void * b) {
    char * a_key = ((bseq1_t*)a)->seq;
    char * b_key = ((bseq1_t*)b)->seq;
    return strncmp(a_key, b_key, 500);
}


/**
 * @brief sort reads lexicographically
 */
void sortReads(bseq1_t *reads, int n_reads){
    qsort(reads, n_reads, sizeof(bseq1_t), compareReads);
}


/* read input from file streams (ks, ks2) and transfer a new batch of seqs into device
	currently only support single-end reads
	read up to actual_chunk_size
	then transfer these new reads to the seq_io on device
	return the number of reads and write new batch of seqs to transfer_data->h_seqs and then transfer_data->d_seqs
 */
static int readInput(kseq_t *ks, kseq_t *ks2, int actual_chunk_size, int copy_comment, transfer_data_t *transfer_data){
	struct timespec timing_start, timing_stop; // variables for printing timings
	resetTransfer(transfer_data);

	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_start);
	int64_t size = 0;
	int n_seqs_read;
	// bseq_read2(actual_chunk_size, &n_seqs_read, ks, ks2, transfer_data);	// this will write to transfer_data
	bseq1_t *seqs = transfer_data->h_seqs;
	if (n_seqs_read == 0) {
		return 0;
	}
	if (copy_comment)
		for (int i = 0; i<n_seqs_read; ++i) {
			seqs[i].comment = 0;
		}
	int max_l_seq = 0;
	for (int i = 0; i<n_seqs_read; ++i){
		size += seqs[i].l_seq;
		max_l_seq = seqs[i].l_seq>max_l_seq ? seqs[i].l_seq : max_l_seq;
	}

	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_stop);
	if (bwa_verbose >= 3) fprintf(stderr, "[M::%-25s] ***load reads from disk took %lu ms\n", __func__, (timing_stop.tv_nsec - timing_start.tv_nsec) / 1000000);
	
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%-25s] *** read sequences %'ld - %'ld (%ld bp)\n", __func__, transfer_data->total_input, transfer_data->total_input+n_seqs_read-1, (long)size);

	// sort reads
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_start);
	sortReads(seqs, n_seqs_read);
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_stop);
	if (bwa_verbose >= 3) fprintf(stderr, "[M::%-25s] ***sort reads took %lu ms\n", __func__, (timing_stop.tv_nsec - timing_start.tv_nsec) / 1000000);

	// transfer to GPU
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_start);
	transfer_data->n_seqs = n_seqs_read;
	convert2DevAddr(seqs, n_seqs_read, transfer_data);
	CUDATransferSeqsIn(transfer_data);
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_stop);
	if (bwa_verbose >= 3) fprintf(stderr, "[M::%-25s] ***transfer to GPU took %lu ms\n", __func__, (timing_stop.tv_nsec - timing_start.tv_nsec) / 1000000);
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%-25s] *** transferred sequences %'ld - %'ld to device\n", __func__, transfer_data->total_input, transfer_data->total_input+n_seqs_read-1);
	transfer_data->total_input += n_seqs_read;
	
	return n_seqs_read;
}

/* output the previous batch of reads 
	first transfer device's seqio to host's seq_io
	then write from host's seq_io to output
 */
static void writeOutput(bool first_batch, transfer_data_t *transfer_data){
	if (first_batch) return;
	if (transfer_data->n_seqs==0) return;

	struct timespec timing_start, timing_stop; // variables for printing timings
	// transfer from device's to host's
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_start);
	CUDATransferSamOut(transfer_data);
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_stop);
	if (bwa_verbose >= 3) fprintf(stderr, "[M::%-25s] ***transfer SAMs to host took %lu ms\n", __func__, (timing_stop.tv_nsec - timing_start.tv_nsec) / 1000000);

	// write from host's seq_io to output
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_start);
	int n_seqs = transfer_data->n_seqs;
	bseq1_t *seqs = transfer_data->h_seqs;
	for (int i = 0; i < n_seqs; ++i)
		if (seqs[i].sam) err_fputs(seqs[i].sam, stdout);
	if (bwa_verbose >= 3) clock_gettime(CLOCK_MONOTONIC_RAW, &timing_stop);
	if (bwa_verbose >= 3) fprintf(stderr, "[M::%-25s] ***write SAMs to stdout took %lu ms\n", __func__, (timing_stop.tv_nsec - timing_start.tv_nsec) / 1000000);
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%-25s] *** wrote output for sequences %'ld - %'ld\n", __func__, transfer_data->total_output, transfer_data->total_output+n_seqs);
	transfer_data->total_output += n_seqs;
	fprintf(stderr, "[M::%-25s] finised %'ld read\n", __func__, transfer_data->total_output);
}



/* processing that hides IO 
    Async in a kernel:
		- transfer processed seqs on device -> host's seqs_io (if there is any)    - process reads stored on seqs_process
		- write from seqs_io to output
		- read new batch of seqs from disk to seqs_io
		- transfer from seqs_io to device
		----------   async wait for both streams -----------
		swap seqs_process <-> seqs_io
		swap corresponding 2 sets of pointers on device
 */
void processHideIO(ktp_aux_t *aux){
	process_data_t *process_data = newProcess(aux->opt, aux->pes0, aux->idx->bwt, aux->idx->bns, aux->idx->pac, aux->kmerHashTab);
	transfer_data_t *transfer_data = newTransfer();
	bool first_batch = true;
	while (first_batch || transfer_data->n_seqs!=0 || process_data->n_seqs!=0 ){		// kernel, process until IO is empty
		if (bwa_verbose>=4) fprintf(stderr, "[M::%-25s] **** seqs in transfer=%'d, seqs in process=%'d \n", __func__, transfer_data->n_seqs, process_data->n_seqs);
		// async output previous batch (process 0)
		auto outputAsync = std::async(std::launch::async, &writeOutput, first_batch, transfer_data); //does nothing if first_batch
			// synchronous version for debug
			// writeOutput(n_seqs_io, first_batch, samSize_io);

		// async process current batch (process 1)
		auto processAsync = std::async(std::launch::async, mem_align_GPU, process_data);	// does nothing if process_data->n_seqs is 0
			// synchronous version for debug
			// samSize_io = mem_align_GPU(aux->gpu_data, preallocated_seqs2, aux->opt, aux->idx->bns);

		// async input next batch (process 0)
		outputAsync.wait();
		auto inputAsync = std::async(std::launch::async, readInput, aux->ks, aux->ks2, aux->actual_chunk_size, aux->copy_comment, transfer_data);
			// synchronous version for debug
			// n_seqs_io = readInput(aux->ks, aux->ks2, aux->actual_chunk_size, aux->copy_comment);

		// swap output of current processed batch to IO, and IO to process for next loop
		// i.e., swap seqs_io <-> seqs_processed on host, and swap pointers on device
		inputAsync.wait();
		processAsync.wait();
			// process_data->n_seqs is now the number of seqs to output in next iteration
			// transfer_data->n_seqs is now the number of seqs to process in next interation
		swapData(process_data, transfer_data);
			// after this swap, process_data->n_seqs is the number of seqs to process in next iteration
			// 					transfer_data->n_seqs is the number of seqs to output in next iteration
		first_batch = false;
		if (bwa_verbose>=4) fprintf(stderr, "[M::%-25s] **** data swapped \n", __func__);
		if (bwa_verbose>=4) fprintf(stderr, "[M::%-25s] **** seqs in transfer=%'d, seqs in process=%'d \n", __func__, transfer_data->n_seqs, process_data->n_seqs);
	}
	return;
}
