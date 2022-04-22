#ifndef CUDASTREAMS_CUH
#define CUDASTREAMS_CUH
#include <stdint.h>
#include "../bwamem.h"
#include "../kmers_index/hashKMerIndex.h"

#ifdef __cplusplus
extern "C"{
#endif

    /* initialize a new instance of process_data_t 
        initialize and transfer constant memory on device:
            - user-defined options 
            - index 
            - memory management (no transfer)
        initialize pinned memory for reads on host
        initialize memory for reads on device
        initialize intermediate processing memory on device
        initialize a cuda stream for processing
     */
	process_data_t* newProcess(
		const mem_opt_t *opt, 
		mem_pestat_t *pes0,
		const bwt_t *bwt, 
		const bntseq_t *bns, 
		const uint8_t *pac,
		const kmers_bucket_t *kmersHashTab
	);

    /* intialize a new set for data transfer (disk <-> CPU <-> GPU)
        intialize pinned memory for reads on host
        initialize memory for reads on device
        initialize a cuda stream for data transfer
     */
    transfer_data_t* newTransfer();

    /* swap reads on the process set and the transfer set
        h_seqs <-> h_seqs (including name, comments, seq, qual)
        d_seqs <-> d_seqs (including name, comments, seq, qual)
    */
    void swapData(process_data_t *process_data, transfer_data_t *transfer_data);

	/* reset a process' data to prepare aligning a new batch
		reset memory management
		reset intermediate data
		reset total SAM size
	 */
	void resetProcess(process_data_t *process_data);

	/* reset transfer data to load in a new batch
		reset name, seq, comment, qual sizes
		reset n_seqs
	 */
	void resetTransfer(transfer_data_t *transfer_data);

	/* 
		transfer seqs from host into device
 	*/
	void CUDATransferSeqsIn(transfer_data_t *transfer_data);

	/* transfer SAM output from device to host
	*/
	void CUDATransferSamOut(transfer_data_t *transfer_data);

#ifdef __cplusplus
}
#endif

#endif