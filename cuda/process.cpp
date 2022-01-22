/* functions that drive the main process (hiding IO, parallel GPUs, etc.) */

#include "process.h"
#include "streams.cuh"
#include <future>


/* read input from file streams (ks, ks2) and transfer a new batch of seqs into device
	currently only support single-end reads
	read up to actual_chunk_size
	then transfer these new reads to the seq_io on device
	return the number of reads and write new batch of seqs to transfer_data->h_seqs and then transfer_data->d_seqs
 */
static int readInput(kseq_t *ks, kseq_t *ks2, int actual_chunk_size, int copy_comment, transfer_data_t *transfer_data){
	int64_t size = 0;
	resetTransfer(transfer_data);
	int n_seqs_read;
	bseq_read2(actual_chunk_size, &n_seqs_read, ks, ks2, transfer_data);	// this will write to transfer_data
	bseq1_t *seqs = transfer_data->h_seqs;
	if (n_seqs_read == 0) {
		return 0;
	}
	if (copy_comment)
		for (int i = 0; i<n_seqs_read; ++i) {
			seqs[i].comment = 0;
		}
	for (int i = 0; i<n_seqs_read; ++i) size += seqs[i].l_seq;
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%-25s] *** read sequences %'ld - %'ld (%ld bp)\n", __func__, transfer_data->total_input, transfer_data->total_input+n_seqs_read-1, (long)size);

	transfer_data->n_seqs = n_seqs_read;
	CUDATransferSeqsIn(transfer_data);
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
	// transfer from device's to host's
	CUDATransferSamOut(transfer_data);
	// write from host's seq_io to output
	int n_seqs = transfer_data->n_seqs;
	bseq1_t *seqs = transfer_data->h_seqs;
	for (int i = 0; i < n_seqs; ++i)
		if (seqs[i].sam) err_fputs(seqs[i].sam, stdout);
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%-25s] *** wrote output for sequences %'ld - %'ld\n", __func__, transfer_data->total_output, transfer_data->total_output+n_seqs);
	fprintf(stderr, "[M::%-25s] finised %'ld read\n", __func__, transfer_data->total_output);
	transfer_data->total_output += n_seqs;
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
	transfer_data_t *transfer_data = newTransfer();
	process_data_t *process_data = newProcess(aux->opt, aux->pes0, aux->idx->bwt, aux->idx->bns, aux->idx->pac);
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
