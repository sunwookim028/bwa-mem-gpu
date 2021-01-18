extern __device__ void bwt_smem_left(const bwt_t *bwt, int len, const uint8_t *q, int x, int min_intv, uint64_t max_intv, int min_seed_len, bwtintv_v *mem);

extern __device__ void bwt_smem_right(const bwt_t *bwt, int len, const uint8_t *q, int x, int min_intv, uint64_t max_intv, int min_seed_len, bwtintv_t *mem_a);

extern __device__ int bwt_smem1a_gpu(const bwt_t *bwt, int len, const uint8_t *q, int x, int min_intv, uint64_t max_intv, bwtintv_v *mem, bwtintv_v *tmpvec[2], void* d_buffer_ptr);

extern __device__ int bwt_seed_strategy1_gpu(const bwt_t *bwt, int len, const uint8_t *q, int x, int min_len, int max_intv, bwtintv_t *mem);

extern __device__ bwtint_t bwt_sa_gpu(const bwt_t *bwt, bwtint_t k);