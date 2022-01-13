/* functions that drive the main process (hiding IO, parallel GPUs, etc.) */

#include "process.h"
#include <future>


/* read input from file streams (ks, ks2) and transfer a new batch of seqs into device
	currently only support single-end reads
	read up to actual_chunk_size
	then transfer these new reads to the seq_io on device
	return the number of reads and write new batch of seqs to preallocated_seqs
 */
static int readInput(kseq_t *ks, kseq_t *ks2, int actual_chunk_size, int copy_comment){
	int64_t size = 0;
	ResetSeqsCounter(); // reset seq counter defined globally
	int n_seqs_read;
	bseq1_t *seqs = bseq_read(actual_chunk_size, &n_seqs_read, ks, ks2);	// this will write to preallocated_seqs
	if (n_seqs_read == 0) {
		return 0;
	}
	if (copy_comment)
		for (int i = 0; i<n_seqs_read; ++i) {
			// free(ret->seqs[i].comment);
			seqs[i].comment = 0;
		}
	for (int i = 0; i<n_seqs_read; ++i) size += seqs[i].l_seq;
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%s] *** read %d sequences (%ld bp)...\n", __func__, n_seqs_read, (long)size);
	
	CUDATransferSeqsIn(n_seqs_read);
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%s] *** transferred %d sequences to device...\n", __func__, n_seqs_read);
	
	return n_seqs_read;
}

/* output the previous batch of reads 
	first transfer device's seqio to host's seq_io
	then write from host's seq_io to output
 */
static void writeOutput(int n_seqs, bool first_batch, int samSize){
	if (first_batch) return;
	if (n_seqs==0) return;
	bseq1_t *seqs = preallocated_seqs;	// host's seqio
	// transfer from device's seq_io to host's seq_io
	CUDATransferSamOut(n_seqs, samSize);
	// write from host's seq_io to output
	for (int i = 0; i < n_seqs; ++i)
		if (seqs[i].sam) err_fputs(seqs[i].sam, stdout);
	if (bwa_verbose >= 3)
		fprintf(stderr, "[M::%s] wrote output for  %'d seqs\n", __func__, n_seqs);
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
	int n_seqs_io = 0;			// number of seqs currently in the allocated memory for IO
	int n_seqs_process = 0;		// number of seqs to be processed
	// io memory and process memory are pre-allocated (preallocated_seqs and preallocated_seqs2 respectively)
	int samSize_io = 0;		// total size of SAM on the device's seqs_io
	bool first_batch = true;
	
	while (first_batch || n_seqs_io!=0 || n_seqs_process!=0 ){		// kernel, process until IO is empty
		// async output previous batch (process 0)
		auto outputAsync = std::async(std::launch::async, &writeOutput, n_seqs_io, first_batch, samSize_io); //does nothing if first_batch
			// synchronous version for debug
			// writeOutput(n_seqs_io, first_batch, samSize_io);

		// async process current batch (process 1)
		prepare_batch_GPU(&aux->gpu_data, preallocated_seqs2, n_seqs_process, aux->opt);	// does nothing if n_seqs is 0
		auto processAsync = std::async(std::launch::async, mem_align_GPU, aux->gpu_data, preallocated_seqs2, aux->opt, aux->idx->bns);	// does nothing if gpu_data->n_seqs is 0
			// synchronous version for debug
			// samSize_io = mem_align_GPU(aux->gpu_data, preallocated_seqs2, aux->opt, aux->idx->bns);
		if (n_seqs_process){
			aux->n_processed += n_seqs_process;
			aux->gpu_data.n_processed += n_seqs_process;
		}

		// async input next batch (process 0)
		outputAsync.wait();
		auto inputAsync = std::async(std::launch::async, readInput, aux->ks, aux->ks2, aux->actual_chunk_size, aux->copy_comment);
			// synchronous version for debug
			// n_seqs_io = readInput(aux->ks, aux->ks2, aux->actual_chunk_size, aux->copy_comment);

		// swap output of current processed batch to IO, and IO to process for next loop
		// i.e., swap seqs_io <-> seqs_processed on host, and swap pointers on device
		n_seqs_io = inputAsync.get();		// number of seqs to process in next iteration
		samSize_io = processAsync.get();	// size of SAM output in next iteration
											// n_seqs_process is now the number of seqs to output in next iteration
		auto tmp = n_seqs_io; n_seqs_io = n_seqs_process; n_seqs_process = tmp; // after this swap, n_seqs_io is the number of seqs to out, n_seqs_process is the number of seqs to process
		SwapPtrs();
		first_batch = false;
		if (bwa_verbose>=4) fprintf(stderr, "[M::%s] **** n_seqs_io=%d, n_seqs_process=%d \n", __func__, n_seqs_io, n_seqs_process);
	}

	return;

}
