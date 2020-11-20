#include "../bwa.h"

extern __device__ __constant__ unsigned char d_nst_nt4_table[256];

int* preprocessing1(bseq1_t* d_seqs, int n_seqs);
