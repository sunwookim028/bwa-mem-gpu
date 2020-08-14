#include <stdint.h>
#include "../bntseq.h"

extern __device__ int bns_pos2rid_gpu(const bntseq_t *bns, int64_t pos_f);
extern __device__ int bns_intv2rid_gpu(const bntseq_t *bns, int64_t rb, int64_t re);
extern __device__ uint8_t *bns_fetch_seq_gpu(const bntseq_t *bns, const uint8_t *pac, int64_t *beg, int64_t mid, int64_t *end, int *rid, void* d_buffer_ptr);
extern __device__ uint8_t *bns_get_seq_gpu(int64_t l_pac, const uint8_t *pac, int64_t beg, int64_t end, int64_t *len, void* d_buffer_ptr);