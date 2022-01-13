#include <stdint.h>
#include "preprocessing.cuh"
#define HASH_LEN 7
#include "errHandler.cuh"

__device__ __constant__ unsigned char d_nst_nt4_table[256] = {
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5 /*'-'*/, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};


/* function to calculate power of base 5*/
__device__ inline int pow5(int x){
	int out = 1;
	for (int i=0; i<x; i++)
		out = out*5;
	return out;
}

/* kernel function to do:
	1. convert seq from char to 2-bit 
	2. hash seq and put hash in buckets
	hash range is from 0 to 5^HASH_LEN
 */
__global__ void hash_kernel(bseq1_t* d_seqs, int n_seqs, 
	int* d_bucket_N, 		// count in each bucket
	int* d_bucket_ids,		// seq_ids storage for buckets
	int bucket_maxlen
	)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x; 		// seq id to process for this thread
	char* seq = d_seqs[i].seq;
	int l_seq = d_seqs[i].l_seq;
	if (i >= n_seqs) return;  // don't run padded threads

	// convert to 2-bit encoding and hash
	int j;
	int hash = 0;
	for (j = 0; j < l_seq; ++j){
		seq[j] = seq[j] < 4? seq[j] : d_nst_nt4_table[(int)seq[j]];
		if (j<HASH_LEN)
			hash += (uint8_t)seq[j]*pow5(j);
	}

	// throw to bucket
	j = atomicAdd(&d_bucket_N[hash], 1); // position in the bucket
	d_bucket_ids[hash*bucket_maxlen+j] = i;
	// check if exceed max bucket len
	if (j+1 > bucket_maxlen){
		printf("bucket %d exceeds limit\n", hash);
		__trap();
	}
}

/* kernel to create hash map from buckets
	threads work on corresponding position on map to find the seq id
 */
 __global__ void hash_map_kernel(int* d_bucket_N, int* d_bucket_ids, int n_bucket, int bucket_maxlen, int* d_hash_map, int n_seqs)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= n_seqs) return;  // don't run padded threads

	// first find the bucket_id
	int sum = 0; int j; int bucket_Nj;
	for (j=0; j<n_bucket; j++){
		bucket_Nj = d_bucket_N[j]; // load this count to register
		sum += bucket_Nj;	// cumulative bucket sum
		if (sum>i){		// we have found the right bucket = j
			sum -= bucket_Nj;	// retrieve the previous sum
			break;
		}
	}
	d_hash_map[i] = d_bucket_ids[j*bucket_maxlen + (i-sum)];
}

/* preprocessing function:
	1. convert all seqs on device to 2-bit
	2. hash seqs and put hash in buckets
	3. create a processing map from threadID to seqID
	return int* d_hash_map with length = n_seqs
 */
int* preprocessing1(bseq1_t* d_seqs, int n_seqs)
{
	// calculate number of hash buckets = 5^HASH_LEN
	int n_bucket = 1;
	for (int i=0; i<HASH_LEN; i++)
		n_bucket = n_bucket*5;
	// allocate buckets and bucket counts
	int* d_bucket_N;
	cudaMalloc((void**)&d_bucket_N, n_bucket*sizeof(int));
	cudaMemset(d_bucket_N, 0, n_bucket*sizeof(int));
	int* d_bucket_ids;
	int bucket_maxlen = n_seqs/n_bucket<<10; // each bucket has length = n_seqs/n_bucket*1024
	gpuErrchk(cudaMalloc((void**)&d_bucket_ids, n_bucket*bucket_maxlen*sizeof(int)));
	// launch kernel to hash and drop to bucket
	dim3 dimGrid(ceil((double)n_seqs/32));
	dim3 dimBlock(32);
	hash_kernel <<< dimGrid, dimBlock, 0 >>> (d_seqs, n_seqs, d_bucket_N, d_bucket_ids, bucket_maxlen);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// allocate hash map
	int* d_hash_map;
	cudaMalloc((void**)&d_hash_map, n_seqs*sizeof(int));
	// launch kernel to calculate map
	hash_map_kernel <<< dimGrid, dimBlock, 0 >>> (d_bucket_N, d_bucket_ids, n_bucket, bucket_maxlen, d_hash_map, n_seqs);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	cudaFree(d_bucket_ids); cudaFree(d_bucket_N);
	return d_hash_map;
}
