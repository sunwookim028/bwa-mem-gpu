#include <stdint.h>

// chaining data struct 
typedef struct {
	int64_t rbeg;
	int32_t qbeg, len;
	int score;
	int This_is_padding_for_CUDA_alignment;
} mem_seed_t; // unaligned memory

typedef struct {
	mem_seed_t *seeds;
	int64_t pos;
	int n, m, first, rid;
	uint32_t w:29, kept:2, is_alt:1;
	float frac_rep;
	// int64_t pos;
} mem_chain_t;
typedef struct { size_t n, m; mem_chain_t *a;  } mem_chain_v;


typedef struct {
	int32_t is_internal:1, n:31;  // n is number of keys in this node
} kbnode_t;

typedef struct {
	kbnode_t *root;
	int	off_key, off_ptr, ilen, elen;
	int	n, t;
	int	n_keys, n_nodes;
} kbtree_chn_t;

#define kb_size(b) ((b)->n_keys)
#define chain_cmp(a, b) (((b).pos < (a).pos) - ((a).pos < (b).pos))
#define	__KB_KEY(x)((mem_chain_t*)((char*)x + 8))
#define __KB_PTR(btr, x)	((kbnode_t**)((char*)x + btr->off_ptr))
extern __device__ kbtree_chn_t *kb_init_chn(int size, void* CUDAKernel_buffer);
extern __device__ void kb_intervalp_chn(kbtree_chn_t *b, const mem_chain_t * __restrict k, mem_chain_t **lower, mem_chain_t **upper);
extern __device__ void kb_putp_chn(kbtree_chn_t *b, const mem_chain_t * __restrict k, void* CUDAKernel_buffer) ;
extern __device__ void __kb_traverse(kbtree_chn_t* b, mem_chain_v* chain, void* CUDAKernel_buffer);