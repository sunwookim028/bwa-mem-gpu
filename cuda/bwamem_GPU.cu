#include "bwamem_GPU.cuh"
#include "CUDAKernel_memmgnt.cuh"
// #include "kvec_CUDA.h"
#include "bwt_CUDA.cuh"
#include "bntseq_CUDA.cuh"
#include "kbtree_CUDA.cuh"
#include "ksort_CUDA.h"
#include "ksw_CUDA.cuh"
#include "bwa_CUDA.cuh"
#include "kstring_CUDA.cuh"
#include <string.h>

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

/* ------------------------------ DEVICE FUNCTIONS TO BE CALLED WITHIN KERNEL ---------------------------*/
/* collection of SA intervals  */
typedef struct {
	bwtintv_v mem, mem1, *tmpv[2];
} smem_aux_t;

/************************
 * Seeding and Chaining *
 ************************/
// return 1 if the seed is merged into the chain
__device__ static int test_and_merge(const mem_opt_t *opt, int64_t l_pac, mem_chain_t *c, const mem_seed_t *p, int seed_rid, void* CUDAKernel_buffer)
{
	int64_t qend, rend, x, y;
	const mem_seed_t *last = &c->seeds[c->n-1];
	qend = last->qbeg + last->len;
	rend = last->rbeg + last->len;
	if (seed_rid != c->rid) return 0; // different chr; request a new chain
	if (p->qbeg >= c->seeds[0].qbeg && p->qbeg + p->len <= qend && p->rbeg >= c->seeds[0].rbeg && p->rbeg + p->len <= rend)
		return 1; // contained seed; do nothing
	if ((last->rbeg < l_pac || c->seeds[0].rbeg < l_pac) && p->rbeg >= l_pac) return 0; // don't chain if on different strand
	x = p->qbeg - last->qbeg; // always non-negtive
	y = p->rbeg - last->rbeg;
	if (y >= 0 && x - y <= opt->w && y - x <= opt->w && x - last->len < opt->max_chain_gap && y - last->len < opt->max_chain_gap) { // grow the chain
		if (c->n == c->m) {
			c->m <<= 1;
			c->seeds = (mem_seed_t*)CUDAKernelRealloc(CUDAKernel_buffer, c->seeds, c->m * sizeof(mem_seed_t), 8);
		}
		c->seeds[c->n++] = *p;
		return 1;
	}
	return 0; // request to add a new chain
}

/* end collection of SA intervals  */

/********************
 * Filtering chains *
 ********************/

#define chn_beg(ch) ((ch).seeds->qbeg)
#define chn_end(ch) ((ch).seeds[(ch).n-1].qbeg + (ch).seeds[(ch).n-1].len)

__device__ int mem_chain_weight(const mem_chain_t *c)
{
	int64_t end;
	int j, w = 0, tmp;
	for (j = 0, end = 0; j < c->n; ++j) {
		const mem_seed_t *s = &c->seeds[j];
		if (s->qbeg >= end) w += s->len;
		else if (s->qbeg + s->len > end) w += s->qbeg + s->len - end;
		end = end > s->qbeg + s->len? end : s->qbeg + s->len;
	}
	tmp = w; w = 0;
	for (j = 0, end = 0; j < c->n; ++j) {
		const mem_seed_t *s = &c->seeds[j];
		if (s->rbeg >= end) w += s->len;
		else if (s->rbeg + s->len > end) w += s->rbeg + s->len - end;
		end = end > s->rbeg + s->len? end : s->rbeg + s->len;
	}
	w = w < tmp? w : tmp;
	return w < 1<<30? w : (1<<30)-1;
}

/*********************************
 * Test if a seed is good enough *
 *********************************/
#define MEM_SHORT_EXT 50
#define MEM_SHORT_LEN 200

#define MEM_HSP_COEF 1.1f
#define MEM_MINSC_COEF 5.5f
#define MEM_SEEDSW_COEF 0.05f

__device__ int mem_seed_sw(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_query, const uint8_t *query, const mem_seed_t *s, void* d_buffer_ptr)
{
	int qb, qe, rid;
	int64_t rb, re, mid, l_pac = bns->l_pac;
	uint8_t *rseq = 0;
	kswr_t x;

	if (s->len >= MEM_SHORT_LEN) return -1; // the seed is longer than the max-extend; no need to do SW
	qb = s->qbeg, qe = s->qbeg + s->len;
	rb = s->rbeg, re = s->rbeg + s->len;
	mid = (rb + re) >> 1;
	qb -= MEM_SHORT_EXT; qb = qb > 0? qb : 0;
	qe += MEM_SHORT_EXT; qe = qe < l_query? qe : l_query;
	rb -= MEM_SHORT_EXT; rb = rb > 0? rb : 0;
	re += MEM_SHORT_EXT; re = re < l_pac<<1? re : l_pac<<1;
	if (rb < l_pac && l_pac < re) {
		if (mid < l_pac) re = l_pac;
		else rb = l_pac;
	}
	if (qe - qb >= MEM_SHORT_LEN || re - rb >= MEM_SHORT_LEN) return -1; // the seed seems good enough; no need to do SW

	rseq = bns_fetch_seq_gpu(bns, pac, &rb, mid, &re, &rid, d_buffer_ptr);
	x = ksw_align2(qe - qb, (uint8_t*)query + qb, re - rb, rseq, 5, opt->mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, KSW_XSTART, 0, d_buffer_ptr);
	// free(rseq);
// printf("unit test 4 x.score = %d\n", x.score);
	return x.score;
}

__device__ void mem_flt_chained_seeds(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_query, const uint8_t *query, int n_chn, mem_chain_t *a, void* d_buffer_ptr)
{
	double min_l = opt->min_chain_weight? MEM_HSP_COEF * opt->min_chain_weight : MEM_MINSC_COEF * log((float)l_query);
	int i, j, k, min_HSP_score = (int)(opt->a * min_l + .499);
	if (min_l > MEM_SEEDSW_COEF * l_query) return; // don't run the following for short reads
	for (i = 0; i < n_chn; ++i) {
		mem_chain_t *c = &a[i];
		for (j = k = 0; j < c->n; ++j) {
			mem_seed_t *s = &c->seeds[j];
			s->score = mem_seed_sw(opt, bns, pac, l_query, query, s, d_buffer_ptr);
			if (s->score < 0 || s->score >= min_HSP_score) {
				s->score = s->score < 0? s->len * opt->a : s->score;
				c->seeds[k++] = *s;
			}
		}
		c->n = k;
	}
}

/****************************************
 * Construct the alignment from a chain *
 ****************************************/

__device__ static inline int cal_max_gap(const mem_opt_t *opt, int qlen)
{
	int l_del = (int)((double)(qlen * opt->a - opt->o_del) / opt->e_del + 1.);
	int l_ins = (int)((double)(qlen * opt->a - opt->o_ins) / opt->e_ins + 1.);
	int l = l_del > l_ins? l_del : l_ins;
	l = l > 1? l : 1;
	return l < opt->w<<1? l : opt->w<<1;
}

#define MAX_BAND_TRY  2

__device__ void mem_chain2aln(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_query, const uint8_t *query, const mem_chain_t *c, mem_alnreg_v *av, void* d_buffer_ptr)
{
	int i, k, rid, max_off[2], aw[2]; // aw: actual bandwidth used in extension
	int64_t l_pac = bns->l_pac, rmax[2], tmp, max = 0;
	const mem_seed_t *s;
	uint8_t *rseq = 0;
	uint64_t *srt;

	if (c->n == 0) return;
	// get the max possible span
	rmax[0] = l_pac<<1; rmax[1] = 0;
	for (i = 0; i < c->n; ++i) {
		int64_t b, e;
		const mem_seed_t *t = &c->seeds[i];
		b = t->rbeg - (t->qbeg + cal_max_gap(opt, t->qbeg));
		e = t->rbeg + t->len + ((l_query - t->qbeg - t->len) + cal_max_gap(opt, l_query - t->qbeg - t->len));
		rmax[0] = rmax[0] < b? rmax[0] : b;
		rmax[1] = rmax[1] > e? rmax[1] : e;
		if (t->len > max) max = t->len;
	}
	rmax[0] = rmax[0] > 0? rmax[0] : 0;
	rmax[1] = rmax[1] < l_pac<<1? rmax[1] : l_pac<<1;
	if (rmax[0] < l_pac && l_pac < rmax[1]) { // crossing the forward-reverse boundary; then choose one side
		if (c->seeds[0].rbeg < l_pac) rmax[1] = l_pac; // this works because all seeds are guaranteed to be on the same strand
		else rmax[0] = l_pac;
	}
	// retrieve the reference sequence
	rseq = bns_fetch_seq_gpu(bns, pac, &rmax[0], c->seeds[0].rbeg, &rmax[1], &rid, d_buffer_ptr);

	srt = (uint64_t*)CUDAKernelMalloc(d_buffer_ptr, c->n * 8, 8);
	for (i = 0; i < c->n; ++i)
		srt[i] = (uint64_t)c->seeds[i].score<<32 | i;
	ks_introsort_64(c->n, srt, d_buffer_ptr);

	for (k = c->n - 1; k >= 0; --k) {
		mem_alnreg_t *a;
		s = &c->seeds[(uint32_t)srt[k]];

		for (i = 0; i < av->n; ++i) { // test whether extension has been made before
			mem_alnreg_t *p = &av->a[i];
			int64_t rd;
			int qd, w, max_gap;
			if (s->rbeg < p->rb || s->rbeg + s->len > p->re || s->qbeg < p->qb || s->qbeg + s->len > p->qe) continue; // not fully contained
			if (s->len - p->seedlen0 > .1 * l_query) continue; // this seed may give a better alignment
			// qd: distance ahead of the seed on query; rd: on reference
			qd = s->qbeg - p->qb; rd = s->rbeg - p->rb;
			max_gap = cal_max_gap(opt, qd < rd? qd : rd); // the maximal gap allowed in regions ahead of the seed
			w = max_gap < p->w? max_gap : p->w; // bounded by the band width
			if (qd - rd < w && rd - qd < w) break; // the seed is "around" a previous hit
			// similar to the previous four lines, but this time we look at the region behind
			qd = p->qe - (s->qbeg + s->len); rd = p->re - (s->rbeg + s->len);
			max_gap = cal_max_gap(opt, qd < rd? qd : rd);
			w = max_gap < p->w? max_gap : p->w;
			if (qd - rd < w && rd - qd < w) break;
		}
		if (i < av->n) { // the seed is (almost) contained in an existing alignment; further testing is needed to confirm it is not leading to a different aln
			for (i = k + 1; i < c->n; ++i) { // check overlapping seeds in the same chain
				const mem_seed_t *t;
				if (srt[i] == 0) continue;
				t = &c->seeds[(uint32_t)srt[i]];
				if (t->len < s->len * .95) continue; // only check overlapping if t is long enough; TODO: more efficient by early stopping
				if (s->qbeg <= t->qbeg && s->qbeg + s->len - t->qbeg >= s->len>>2 && t->qbeg - s->qbeg != t->rbeg - s->rbeg) break;
				if (t->qbeg <= s->qbeg && t->qbeg + t->len - s->qbeg >= s->len>>2 && s->qbeg - t->qbeg != s->rbeg - t->rbeg) break;
			}
			if (i == c->n) { // no overlapping seeds; then skip extension
				srt[k] = 0; // mark that seed extension has not been performed
				continue;
			}
		}

	// 	a = kv_pushp(type=mem_alnreg_t, v=*av);
		a = (((av->n == av->m)?
		    (av->m = (av->m? av->m<<1 : 2),
			av->a = (mem_alnreg_t*)CUDAKernelRealloc(d_buffer_ptr, av->a, sizeof(mem_alnreg_t) * av->m, 8), 0)
		    : 0), &(av->a[av->n++]));
		memset(a, 0, sizeof(mem_alnreg_t));
		a->w = aw[0] = aw[1] = opt->w;
		a->score = a->truesc = -1;
		a->rid = c->rid;

		if (s->qbeg) { // left extension
			uint8_t *rs, *qs;
			int qle, tle, gtle, gscore;
			qs = (uint8_t*)CUDAKernelMalloc(d_buffer_ptr, s->qbeg, 1);
			for (i = 0; i < s->qbeg; ++i) qs[i] = query[s->qbeg - 1 - i];
			tmp = s->rbeg - rmax[0];
			rs = (uint8_t*)CUDAKernelMalloc(d_buffer_ptr, tmp, 1);
			for (i = 0; i < tmp; ++i) rs[i] = rseq[tmp - 1 - i];
			for (i = 0; i < MAX_BAND_TRY; ++i) {
				int prev = a->score;
				aw[0] = opt->w << i;
				a->score = ksw_extend2(s->qbeg, qs, tmp, rs, 5, opt->mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, aw[0], opt->pen_clip5, opt->zdrop, s->len * opt->a, &qle, &tle, &gtle, &gscore, &max_off[0], d_buffer_ptr);
				if (a->score == prev || max_off[0] < (aw[0]>>1) + (aw[0]>>2)) break;
			}
			// check whether we prefer to reach the end of the query
			if (gscore <= 0 || gscore <= a->score - opt->pen_clip5) { // local extension
				a->qb = s->qbeg - qle, a->rb = s->rbeg - tle;
				a->truesc = a->score;
			} else { // to-end extension
				a->qb = 0, a->rb = s->rbeg - gtle;
				a->truesc = gscore;
			}
	// 		free(qs); free(rs);
		} else a->score = a->truesc = s->len * opt->a, a->qb = 0, a->rb = s->rbeg;

		if (s->qbeg + s->len != l_query) { // right extension
			int qle, tle, qe, re, gtle, gscore, sc0 = a->score;
			qe = s->qbeg + s->len;
			re = s->rbeg + s->len - rmax[0];
			for (i = 0; i < MAX_BAND_TRY; ++i) {
				int prev = a->score;
				aw[1] = opt->w << i;
				a->score = ksw_extend2(l_query - qe, query + qe, rmax[1] - rmax[0] - re, rseq + re, 5, opt->mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, aw[1], opt->pen_clip3, opt->zdrop, sc0, &qle, &tle, &gtle, &gscore, &max_off[1], d_buffer_ptr);
				if (a->score == prev || max_off[1] < (aw[1]>>1) + (aw[1]>>2)) break;
			}
			// similar to the above
			if (gscore <= 0 || gscore <= a->score - opt->pen_clip3) { // local extension
				a->qe = qe + qle, a->re = rmax[0] + re + tle;
				a->truesc += a->score - sc0;
			} else { // to-end extension
				a->qe = l_query, a->re = rmax[0] + re + gtle;
				a->truesc += gscore - sc0;
			}
		} else a->qe = l_query, a->re = s->rbeg + s->len;

		// compute seedcov
		for (i = 0, a->seedcov = 0; i < c->n; ++i) {
			const mem_seed_t *t = &c->seeds[i];
			if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe && t->rbeg >= a->rb && t->rbeg + t->len <= a->re) // seed fully contained
				a->seedcov += t->len; // this is not very accurate, but for approx. mapQ, this is good enough
		}
		a->w = aw[0] > aw[1]? aw[0] : aw[1];
		a->seedlen0 = s->len;

		a->frac_rep = c->frac_rep;
	}
	// free(srt); free(rseq);
}


/******************************
 * De-overlap single-end hits *
 ******************************/
#define PATCH_MAX_R_BW 0.05f
#define PATCH_MIN_SC_RATIO 0.90f

__device__ int mem_patch_reg(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, uint8_t *query, const mem_alnreg_t *a, const mem_alnreg_t *b, int *_w, void* d_buffer_ptr)
{
	int w, score, q_s, r_s;
	double r;
	if (bns == 0 || pac == 0 || query == 0) return 0;
	if (a->rb < bns->l_pac && b->rb >= bns->l_pac) return 0; // on different strands
	if (a->qb >= b->qb || a->qe >= b->qe || a->re >= b->re) return 0; // not colinear
	w = (a->re - b->rb) - (a->qe - b->qb); // required bandwidth
	w = w > 0? w : -w; // l = abs(l)
	r = (double)(a->re - b->rb) / (b->re - a->rb) - (double)(a->qe - b->qb) / (b->qe - a->qb); // relative bandwidth
	r = r > 0.? r : -r; // r = fabs(r)

	if (a->re < b->rb || a->qe < b->qb) { // no overlap on query or on ref
		if (w > opt->w<<1 || r >= PATCH_MAX_R_BW) return 0; // the bandwidth or the relative bandwidth is too large
	} else if (w > opt->w<<2 || r >= PATCH_MAX_R_BW*2) return 0; // more permissive if overlapping on both ref and query
	// global alignment
	w += a->w + b->w;
	w = w < opt->w<<2? w : opt->w<<2;
	bwa_gen_cigar2_gpu(opt->mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, w, bns->l_pac, pac, b->qe - a->qb, query + a->qb, a->rb, b->re, &score, 0, 0, d_buffer_ptr);
	q_s = (int)((double)(b->qe - a->qb) / ((b->qe - b->qb) + (a->qe - a->qb)) * (b->score + a->score) + .499); // predicted score from query
	r_s = (int)((double)(b->re - a->rb) / ((b->re - b->rb) + (a->re - a->rb)) * (b->score + a->score) + .499); // predicted score from ref
	if ((double)score / (q_s > r_s? q_s : r_s) < PATCH_MIN_SC_RATIO) return 0;
	*_w = w;
	return score;
}

__device__ int mem_sort_dedup_patch(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, uint8_t *query, int n, mem_alnreg_t *a, void* d_buffer_ptr)
{
	int m, i, j;
	if (n <= 1) return n;
	ks_introsort_mem_ars2(n, a, d_buffer_ptr); // sort by the END position, not START!
	for (i = 0; i < n; ++i) a[i].n_comp = 1;
	for (i = 1; i < n; ++i) {
		mem_alnreg_t *p = &a[i];
		if (p->rid != a[i-1].rid || p->rb >= a[i-1].re + opt->max_chain_gap) continue; // then no need to go into the loop below
		for (j = i - 1; j >= 0 && p->rid == a[j].rid && p->rb < a[j].re + opt->max_chain_gap; --j) {
			mem_alnreg_t *q = &a[j];
			int64_t orr, oq, mr, mq;
			int score, w;
			if (q->qe == q->qb) continue; // a[j] has been excluded
			orr = q->re - p->rb; // overlap length on the reference
			oq = q->qb < p->qb? q->qe - p->qb : p->qe - q->qb; // overlap length on the query
			mr = q->re - q->rb < p->re - p->rb? q->re - q->rb : p->re - p->rb; // min ref len in alignment
			mq = q->qe - q->qb < p->qe - p->qb? q->qe - q->qb : p->qe - p->qb; // min qry len in alignment
			if (orr > opt->mask_level_redun * mr && oq > opt->mask_level_redun * mq) { // one of the hits is redundant
				if (p->score < q->score) {
					p->qe = p->qb;
					break;
				} else q->qe = q->qb;
			} else if (q->rb < p->rb && (score = mem_patch_reg(opt, bns, pac, query, q, p, &w, d_buffer_ptr)) > 0) { // then merge q into p
				p->n_comp += q->n_comp + 1;
				p->seedcov = p->seedcov > q->seedcov? p->seedcov : q->seedcov;
				p->sub = p->sub > q->sub? p->sub : q->sub;
				p->csub = p->csub > q->csub? p->csub : q->csub;
				p->qb = q->qb, p->rb = q->rb;
				p->truesc = p->score = score;
				p->w = w;
				q->qb = q->qe;
			}
		}
	}
	for (i = 0, m = 0; i < n; ++i) // exclude identical hits
		if (a[i].qe > a[i].qb) {
			if (m != i) a[m++] = a[i];
			else ++m;
		}
	n = m;
	ks_introsort_mem_ars2(n, a, d_buffer_ptr);
	for (i = 1; i < n; ++i) { // mark identical hits
		if (a[i].score == a[i-1].score && a[i].rb == a[i-1].rb && a[i].qb == a[i-1].qb)
			a[i].qe = a[i].qb;
	}
	for (i = 1, m = 1; i < n; ++i) // exclude identical hits
		if (a[i].qe > a[i].qb) {
			if (m != i) a[m++] = a[i];
			else ++m;
		}
	return m;
}


/********************************************
 * Infer Insert-size distribution from data *
 ********************************************/

#define MIN_RATIO     0.8
#define MIN_DIR_CNT   10
#define MIN_DIR_RATIO 0.05
#define OUTLIER_BOUND 2.0
#define MAPPING_BOUND 3.0
#define MAX_STDDEV    4.0

typedef struct { size_t n, m; uint64_t *a; } uint64_v;


__device__ static inline int mem_infer_dir(int64_t l_pac, int64_t b1, int64_t b2, int64_t *dist)
{
	int64_t p2;
	int r1 = (b1 >= l_pac), r2 = (b2 >= l_pac);
	p2 = r1 == r2? b2 : (l_pac<<1) - 1 - b2; // p2 is the coordinate of read 2 on the read 1 strand
	*dist = p2 > b1? p2 - b1 : b1 - p2;
	return (r1 == r2? 0 : 1) ^ (p2 > b1? 0 : 3);
}

__device__ static int cal_sub(const mem_opt_t *opt, mem_alnreg_v *r)
{
	int j;
	for (j = 1; j < r->n; ++j) { // choose unique alignment
		int b_max = r->a[j].qb > r->a[0].qb? r->a[j].qb : r->a[0].qb;
		int e_min = r->a[j].qe < r->a[0].qe? r->a[j].qe : r->a[0].qe;
		if (e_min > b_max) { // have overlap
			int min_l = r->a[j].qe - r->a[j].qb < r->a[0].qe - r->a[0].qb? r->a[j].qe - r->a[j].qb : r->a[0].qe - r->a[0].qb;
			if (e_min - b_max >= min_l * opt->mask_level) break; // significant overlap
		}
	}
	return j < r->n? r->a[j].score : opt->min_seed_len * opt->a;
}

__device__ void mem_pestat_GPU(const mem_opt_t *opt, int64_t l_pac, int n, const mem_alnreg_v *regs, mem_pestat_t pes[4], void* d_buffer_ptr)
{
	int i, d, max;
	uint64_v isize[4];
	memset(pes, 0, 4 * sizeof(mem_pestat_t));
	memset(isize, 0, 4 * sizeof(uint64_v));
	for (i = 0; i < n>>1; ++i) {
		int dir;
		int64_t is;
		mem_alnreg_v *r[2];
		r[0] = (mem_alnreg_v*)&regs[i<<1|0];
		r[1] = (mem_alnreg_v*)&regs[i<<1|1];
		if (r[0]->n == 0 || r[1]->n == 0) continue;
		if (cal_sub(opt, r[0]) > MIN_RATIO * r[0]->a[0].score) continue;
		if (cal_sub(opt, r[1]) > MIN_RATIO * r[1]->a[0].score) continue;
		if (r[0]->a[0].rid != r[1]->a[0].rid) continue; // not on the same chr
		dir = mem_infer_dir(l_pac, r[0]->a[0].rb, r[1]->a[0].rb, &is);
		if (is && is <= opt->max_ins) {
			// kv_push(uint64_t, v=isize[dir], x=is);
			if (isize[dir].n == isize[dir].m) {
				isize[dir].m = isize[dir].m? isize[dir].m<<1 : 2;
				isize[dir].a = (uint64_t*)CUDAKernelRealloc(d_buffer_ptr, isize[dir].a, sizeof(uint64_t) * isize[dir].m, 8);
			}
			isize[dir].a[isize[dir].n++] = is;
		}
	}
	for (d = 0; d < 4; ++d) { // TODO: this block is nearly identical to the one in bwtsw2_pair.c. It would be better to merge these two.
		mem_pestat_t *r = &pes[d];
		uint64_v *q = &isize[d];
		int p25, p50, p75, x;
		if (q->n < MIN_DIR_CNT) {
			r->failed = 1;
			// free(q->a);
			continue;
		}
		ks_introsort_64(q->n, q->a, d_buffer_ptr);
		p25 = q->a[(int)(.25 * q->n + .499)];
		p50 = q->a[(int)(.50 * q->n + .499)];
		p75 = q->a[(int)(.75 * q->n + .499)];
		r->low  = (int)(p25 - OUTLIER_BOUND * (p75 - p25) + .499);
		if (r->low < 1) r->low = 1;
		r->high = (int)(p75 + OUTLIER_BOUND * (p75 - p25) + .499);
		for (i = x = 0, r->avg = 0; i < q->n; ++i)
			if (q->a[i] >= r->low && q->a[i] <= r->high)
				r->avg += q->a[i], ++x;
		r->avg /= x;
		for (i = 0, r->std = 0; i < q->n; ++i)
			if (q->a[i] >= r->low && q->a[i] <= r->high)
				r->std += (q->a[i] - r->avg) * (q->a[i] - r->avg);
		r->std = sqrt(r->std / x);
		r->low  = (int)(p25 - MAPPING_BOUND * (p75 - p25) + .499);
		r->high = (int)(p75 + MAPPING_BOUND * (p75 - p25) + .499);
		if (r->low  > r->avg - MAX_STDDEV * r->std) r->low  = (int)(r->avg - MAX_STDDEV * r->std + .499);
		if (r->high < r->avg + MAX_STDDEV * r->std) r->high = (int)(r->avg + MAX_STDDEV * r->std + .499);
		if (r->low < 1) r->low = 1;
		// free(q->a);
	}
	for (d = 0, max = 0; d < 4; ++d)
		max = max > isize[d].n? max : isize[d].n;
	for (d = 0; d < 4; ++d)
		if (pes[d].failed == 0 && isize[d].n < max * MIN_DIR_RATIO) {
			pes[d].failed = 1;
		}
}


/*****************************
 * Basic hit->SAM conversion *
 *****************************/

__device__ static inline int infer_bw(int l1, int l2, int score, int a, int q, int r)
{
	int w;
	if (l1 == l2 && l1 * a - score < (q + r - a)<<1) return 0; // to get equal alignment length, we need at least two gaps
	w = ((double)((l1 < l2? l1 : l2) * a - score - q) / r + 2.);
	if (w < abs(l1 - l2)) w = abs(l1 - l2);
	return w;
}

__device__ static inline int get_rlen(int n_cigar, const uint32_t *cigar)
{
	int k, l;
	for (k = l = 0; k < n_cigar; ++k) {
		int op = cigar[k]&0xf;
		if (op == 0 || op == 2)
			l += cigar[k]>>4;
	}
	return l;
}

__device__ static inline void add_cigar(const mem_opt_t *opt, mem_aln_t *p, kstring_t *str, int which, void* d_buffer_ptr)
{
	int i;
	if (p->n_cigar) { // aligned
		for (i = 0; i < p->n_cigar; ++i) {
			int c = p->cigar[i]&0xf;
			if (!(opt->flag&MEM_F_SOFTCLIP) && !p->is_alt && (c == 3 || c == 4))
				c = which? 4 : 3; // use hard clipping for supplementary alignments
			kputw(p->cigar[i]>>4, str, d_buffer_ptr); kputc("MIDSH"[c], str, d_buffer_ptr);
		}
	} else kputc('*', str, d_buffer_ptr); // having a coordinate but unaligned (e.g. when copy_mate is true)
}

__device__ static void mem_aln2sam(const mem_opt_t *opt, const bntseq_t *bns, kstring_t *str, bseq1_t *s, int n, const mem_aln_t *list, int which, const mem_aln_t *m_, void* d_buffer_ptr)
{
	int i, l_name;
	mem_aln_t ptmp = list[which], *p = &ptmp, mtmp, *m = 0; // make a copy of the alignment to convert

	if (m_) mtmp = *m_, m = &mtmp;
	// set flag
	p->flag |= m? 0x1 : 0; // is paired in sequencing
	p->flag |= p->rid < 0? 0x4 : 0; // is mapped
	p->flag |= m && m->rid < 0? 0x8 : 0; // is mate mapped
	if (p->rid < 0 && m && m->rid >= 0) // copy mate to alignment
		p->rid = m->rid, p->pos = m->pos, p->is_rev = m->is_rev, p->n_cigar = 0;
	if (m && m->rid < 0 && p->rid >= 0) // copy alignment to mate
		m->rid = p->rid, m->pos = p->pos, m->is_rev = p->is_rev, m->n_cigar = 0;
	p->flag |= p->is_rev? 0x10 : 0; // is on the reverse strand
	p->flag |= m && m->is_rev? 0x20 : 0; // is mate on the reverse strand

	// print up to CIGAR
	l_name = strlen_GPU(s->name);
	ks_resize(str, str->l + s->l_seq + l_name + (s->qual? s->l_seq : 0) + 20, d_buffer_ptr);
	kputsn(s->name, l_name, str, d_buffer_ptr); kputc('\t', str, d_buffer_ptr); // QNAME
	kputw((p->flag&0xffff) | (p->flag&0x10000? 0x100 : 0), str, d_buffer_ptr); kputc('\t', str, d_buffer_ptr); // FLAG
	if (p->rid >= 0) { // with coordinate
		kputs(bns->anns[p->rid].name, str, d_buffer_ptr); kputc('\t', str, d_buffer_ptr); // RNAME
		kputl(p->pos + 1, str, d_buffer_ptr); kputc('\t', str, d_buffer_ptr); // POS
		kputw(p->mapq, str, d_buffer_ptr); kputc('\t', str, d_buffer_ptr); // MAPQ
		add_cigar(opt, p, str, which, d_buffer_ptr);
	} else kputsn("*\t0\t0\t*", 7, str, d_buffer_ptr); // without coordinte
	kputc('\t', str, d_buffer_ptr);

	// print the mate position if applicable
	if (m && m->rid >= 0) {
		if (p->rid == m->rid) kputc('=', str, d_buffer_ptr);
		else kputs(bns->anns[m->rid].name, str, d_buffer_ptr);
		kputc('\t', str, d_buffer_ptr);
		kputl(m->pos + 1, str, d_buffer_ptr); kputc('\t', str, d_buffer_ptr);
		if (p->rid == m->rid) {
			int64_t p0 = p->pos + (p->is_rev? get_rlen(p->n_cigar, p->cigar) - 1 : 0);
			int64_t p1 = m->pos + (m->is_rev? get_rlen(m->n_cigar, m->cigar) - 1 : 0);
			if (m->n_cigar == 0 || p->n_cigar == 0) kputc('0', str, d_buffer_ptr);
			else kputl(-(p0 - p1 + (p0 > p1? 1 : p0 < p1? -1 : 0)), str, d_buffer_ptr);
		} else kputc('0', str, d_buffer_ptr);
	} else kputsn("*\t0\t0", 5, str, d_buffer_ptr);
	kputc('\t', str, d_buffer_ptr);

	// print SEQ and QUAL
	if (p->flag & 0x100) { // for secondary alignments, don't write SEQ and QUAL
		kputsn("*\t*", 3, str, d_buffer_ptr);
	} else if (!p->is_rev) { // the forward strand
		int i, qb = 0, qe = s->l_seq;
		if (p->n_cigar && which && !(opt->flag&MEM_F_SOFTCLIP) && !p->is_alt) { // have cigar && not the primary alignment && not softclip all
			if ((p->cigar[0]&0xf) == 4 || (p->cigar[0]&0xf) == 3) qb += p->cigar[0]>>4;
			if ((p->cigar[p->n_cigar-1]&0xf) == 4 || (p->cigar[p->n_cigar-1]&0xf) == 3) qe -= p->cigar[p->n_cigar-1]>>4;
		}
		ks_resize(str, str->l + (qe - qb) + 1, d_buffer_ptr);
		for (i = qb; i < qe; ++i) str->s[str->l++] = "ACGTN"[(int)s->seq[i]];
		kputc('\t', str, d_buffer_ptr);
		if (s->qual) { // printf qual
			ks_resize(str, str->l + (qe - qb) + 1, d_buffer_ptr);
			for (i = qb; i < qe; ++i) str->s[str->l++] = s->qual[i];
			str->s[str->l] = 0;
		} else kputc('*', str, d_buffer_ptr);
	} else { // the reverse strand
		int i, qb = 0, qe = s->l_seq;
		if (p->n_cigar && which && !(opt->flag&MEM_F_SOFTCLIP) && !p->is_alt) {
			if ((p->cigar[0]&0xf) == 4 || (p->cigar[0]&0xf) == 3) qe -= p->cigar[0]>>4;
			if ((p->cigar[p->n_cigar-1]&0xf) == 4 || (p->cigar[p->n_cigar-1]&0xf) == 3) qb += p->cigar[p->n_cigar-1]>>4;
		}
		ks_resize(str, str->l + (qe - qb) + 1, d_buffer_ptr);
		for (i = qe-1; i >= qb; --i) str->s[str->l++] = "TGCAN"[(int)s->seq[i]];
		kputc('\t', str, d_buffer_ptr);
		if (s->qual) { // printf qual
			ks_resize(str, str->l + (qe - qb) + 1, d_buffer_ptr);
			for (i = qe-1; i >= qb; --i) str->s[str->l++] = s->qual[i];
			str->s[str->l] = 0;
		} else kputc('*', str, d_buffer_ptr);
	}

	// print optional tags
	if (p->n_cigar) {
		kputsn("\tNM:i:", 6, str, d_buffer_ptr); kputw(p->NM, str, d_buffer_ptr);
		kputsn("\tMD:Z:", 6, str, d_buffer_ptr); kputs((char*)(p->cigar + p->n_cigar), str, d_buffer_ptr);
	}
	if (m && m->n_cigar) { kputsn("\tMC:Z:", 6, str, d_buffer_ptr); add_cigar(opt, m, str, which, d_buffer_ptr); }
	if (p->score >= 0) { kputsn("\tAS:i:", 6, str, d_buffer_ptr); kputw(p->score, str, d_buffer_ptr); }
	if (p->sub >= 0) { kputsn("\tXS:i:", 6, str, d_buffer_ptr); kputw(p->sub, str, d_buffer_ptr); }
	// if (bwa_rg_id[0]) { kputsn("\tRG:Z:", 6, str, d_buffer_ptr); kputs(bwa_rg_id, str, d_buffer_ptr); }
	if (!(p->flag & 0x100)) { // not multi-hit
		for (i = 0; i < n; ++i)
			if (i != which && !(list[i].flag&0x100)) break;
		if (i < n) { // there are other primary hits; output them
			kputsn("\tSA:Z:", 6, str, d_buffer_ptr);
			for (i = 0; i < n; ++i) {
				const mem_aln_t *r = &list[i];
				int k;
				if (i == which || (r->flag&0x100)) continue; // proceed if: 1) different from the current; 2) not shadowed multi hit
				kputs(bns->anns[r->rid].name, str, d_buffer_ptr); kputc(',', str, d_buffer_ptr);
				kputl(r->pos+1, str, d_buffer_ptr); kputc(',', str, d_buffer_ptr);
				kputc("+-"[r->is_rev], str, d_buffer_ptr); kputc(',', str, d_buffer_ptr);
				for (k = 0; k < r->n_cigar; ++k) {
					kputw(r->cigar[k]>>4, str, d_buffer_ptr); kputc("MIDSH"[r->cigar[k]&0xf], str, d_buffer_ptr);
				}
				kputc(',', str, d_buffer_ptr); kputw(r->mapq, str, d_buffer_ptr);
				kputc(',', str, d_buffer_ptr); kputw(r->NM, str, d_buffer_ptr);
				kputc(';', str, d_buffer_ptr);
			}
		}
		if (p->alt_sc > 0)
			ksprintf(str, "\tpa:f:%.3f", (float)p->score / p->alt_sc, d_buffer_ptr);
	}
	if (p->XA) {
		kputsn((opt->flag&MEM_F_XB)? "\tXB:Z:" : "\tXA:Z:", 6, str, d_buffer_ptr);
		kputs(p->XA, str, d_buffer_ptr);
	}
	if (s->comment) { kputc('\t', str, d_buffer_ptr); kputs(s->comment, str, d_buffer_ptr); }
	if ((opt->flag&MEM_F_REF_HDR) && p->rid >= 0 && bns->anns[p->rid].anno != 0 && bns->anns[p->rid].anno[0] != 0) {
		int tmp;
		kputsn("\tXR:Z:", 6, str, d_buffer_ptr);
		tmp = str->l;
		kputs(bns->anns[p->rid].anno, str, d_buffer_ptr);
		for (i = tmp; i < str->l; ++i) // replace TAB in the comment to SPACE
			if (str->s[i] == '\t') str->s[i] = ' ';
	}
	kputc('\n', str, d_buffer_ptr);
}


/*****************************************************
 * Device functions for generating alignment results *
 *****************************************************/
typedef struct { size_t n, m; int *a; } int_v;

__device__ static inline int64_t bns_depos_GPU(const bntseq_t *bns, int64_t pos, int *is_rev)
{
	return (*is_rev = (pos >= bns->l_pac))? (bns->l_pac<<1) - 1 - pos : pos;
}

__device__ static int mem_approx_mapq_se(const mem_opt_t *opt, const mem_alnreg_t *a)
{
	int mapq, l, sub = a->sub? a->sub : opt->min_seed_len * opt->a;
	double identity;
	sub = a->csub > sub? a->csub : sub;
	if (sub >= a->score) return 0;
	l = a->qe - a->qb > a->re - a->rb? a->qe - a->qb : a->re - a->rb;
	identity = 1. - (double)(l * opt->a - a->score) / (opt->a + opt->b) / l;
	if (a->score == 0) {
		mapq = 0;
	} else if (opt->mapQ_coef_len > 0) {
		double tmp;
		tmp = l < opt->mapQ_coef_len? 1. : opt->mapQ_coef_fac / log((float)l);
		tmp *= identity * identity;
		mapq = (int)(6.02 * (a->score - sub) / opt->a * tmp * tmp + .499);
	} else {
		mapq = (int)(MEM_MAPQ_COEF * (1. - (double)sub / a->score) * log((float)a->seedcov) + .499);
		mapq = identity < 0.95? (int)(mapq * identity * identity + .499) : mapq;
	}
	if (a->sub_n > 0) mapq -= (int)(4.343 * log((float)a->sub_n+1) + .499);
	if (mapq > 60) mapq = 60;
	if (mapq < 0) mapq = 0;
	mapq = (int)(mapq * (1. - a->frac_rep) + .499);
	return mapq;
}


__device__ static inline uint64_t hash_64(uint64_t key)
{
	key += ~(key << 32);
	key ^= (key >> 22);
	key += ~(key << 13);
	key ^= (key >> 8);
	key += (key << 3);
	key ^= (key >> 15);
	key += ~(key << 27);
	key ^= (key >> 31);
	return key;
}

__device__ static void mem_mark_primary_se_core_GPU(const mem_opt_t *opt, int n, mem_alnreg_t *a, int_v *z, void* d_buffer_ptr)
{ // similar to the loop in mem_chain_flt()
	int i, k, tmp;
	tmp = opt->a + opt->b;
	tmp = opt->o_del + opt->e_del > tmp? opt->o_del + opt->e_del : tmp;
	tmp = opt->o_ins + opt->e_ins > tmp? opt->o_ins + opt->e_ins : tmp;
	z->n = 0;
	// kv_push(type=int, v=*z, x=0);
	if (z->n == z->m) {
		z->m = z->m? z->m<<1 : 2;
		z->a = (int*)CUDAKernelRealloc(d_buffer_ptr, z->a, sizeof(int) * z->m, 4);
	}
	z->a[z->n++] = 0;	
	for (i = 1; i < n; ++i) {
		for (k = 0; k < z->n; ++k) {
			int j = z->a[k];
			int b_max = a[j].qb > a[i].qb? a[j].qb : a[i].qb;
			int e_min = a[j].qe < a[i].qe? a[j].qe : a[i].qe;
			if (e_min > b_max) { // have overlap
				int min_l = a[i].qe - a[i].qb < a[j].qe - a[j].qb? a[i].qe - a[i].qb : a[j].qe - a[j].qb;
				if (e_min - b_max >= min_l * opt->mask_level) { // significant overlap
					if (a[j].sub == 0) a[j].sub = a[i].score;
					if (a[j].score - a[i].score <= tmp && (a[j].is_alt || !a[i].is_alt))
						++a[j].sub_n;
					break;
				}
			}
		}
		if (k == z->n){
			// kv_push(int, *z, i);
			if (z->n == z->m) {
				z->m = z->m? z->m<<1 : 2;
				z->a = (int*)CUDAKernelRealloc(d_buffer_ptr, z->a, sizeof(int) * z->m, 4);
			}
			z->a[z->n++] = i;
		}
		else a[i].secondary = z->a[k];
	}
}

__device__ static mem_aln_t mem_reg2aln_GPU(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_query, const char *query_, const mem_alnreg_t *ar, void* d_buffer_ptr)
{
	mem_aln_t a;
	int i, w2, tmp, qb, qe, NM, score, is_rev, last_sc = -(1<<30), l_MD;
	int64_t pos, rb, re;
	uint8_t *query;

	memset(&a, 0, sizeof(mem_aln_t));
	if (ar == 0 || ar->rb < 0 || ar->re < 0) { // generate an unmapped record
		a.rid = -1; a.pos = -1; a.flag |= 0x4;
		return a;
	}
	qb = ar->qb, qe = ar->qe;
	rb = ar->rb, re = ar->re;
	query = (uint8_t*)CUDAKernelMalloc(d_buffer_ptr, l_query, 1);
	for (i = 0; i < l_query; ++i) // convert to the nt4 encoding
		query[i] = query_[i] < 5? query_[i] : d_nst_nt4_table[(int)query_[i]];
	a.mapq = ar->secondary < 0? mem_approx_mapq_se(opt, ar) : 0;
	if (ar->secondary >= 0) a.flag |= 0x100; // secondary alignment
	tmp = infer_bw(qe - qb, re - rb, ar->truesc, opt->a, opt->o_del, opt->e_del);
	w2  = infer_bw(qe - qb, re - rb, ar->truesc, opt->a, opt->o_ins, opt->e_ins);
	w2 = w2 > tmp? w2 : tmp;
	if (w2 > opt->w) w2 = w2 < ar->w? w2 : ar->w;
	i = 0; a.cigar = 0;
	do {
		// free(a.cigar);
		w2 = w2 < opt->w<<2? w2 : opt->w<<2;
		a.cigar = bwa_gen_cigar2_gpu(opt->mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, w2, bns->l_pac, pac, qe - qb, (uint8_t*)&query[qb], rb, re, &score, &a.n_cigar, &NM, d_buffer_ptr);
		if (score == last_sc || w2 == opt->w<<2) break; // it is possible that global alignment and local alignment give different scores
		last_sc = score;
		w2 <<= 1;
	} while (++i < 3 && score < ar->truesc - opt->a);
	l_MD = strlen_GPU((char*)(a.cigar + a.n_cigar)) + 1;

	a.NM = NM;
	pos = bns_depos_GPU(bns, rb < bns->l_pac? rb : re - 1, &is_rev);
	a.is_rev = is_rev;
	if (a.n_cigar > 0) { // squeeze out leading or trailing deletions
		if ((a.cigar[0]&0xf) == 2) {
			pos += a.cigar[0]>>4;
			--a.n_cigar;
			cudaKernelMemmove(a.cigar + 1, a.cigar, a.n_cigar * 4 + l_MD);
		} else if ((a.cigar[a.n_cigar-1]&0xf) == 2) {
			--a.n_cigar;
			cudaKernelMemmove(a.cigar + a.n_cigar + 1, a.cigar + a.n_cigar, l_MD); // MD needs to be moved accordingly
		}
	}
	if (qb != 0 || qe != l_query) { // add clipping to CIGAR
		int clip5, clip3;
		clip5 = is_rev? l_query - qe : qb;
		clip3 = is_rev? qb : l_query - qe;
		a.cigar = (uint32_t*)CUDAKernelRealloc(d_buffer_ptr, a.cigar, 4 * (a.n_cigar + 2) + l_MD, 4);
		if (clip5) {
			cudaKernelMemmove(a.cigar, a.cigar+1, a.n_cigar * 4 + l_MD); // make room for 5'-end clipping
			a.cigar[0] = clip5<<4 | 3;
			++a.n_cigar;
		}
		if (clip3) {
			cudaKernelMemmove(a.cigar + a.n_cigar, a.cigar + a.n_cigar + 1, l_MD); // make room for 3'-end clipping
			a.cigar[a.n_cigar++] = clip3<<4 | 3;
		}
	}
	a.rid = bns_pos2rid_gpu(bns, pos);
	// assert(a.rid == ar->rid);
	a.pos = pos - bns->anns[a.rid].offset;
	a.score = ar->score; a.sub = ar->sub > ar->csub? ar->sub : ar->csub;
	a.is_alt = ar->is_alt; a.alt_sc = ar->alt_sc;
	// free(query);
	return a;
}


__device__ int mem_mark_primary_se_GPU(const mem_opt_t *d_opt, int n, mem_alnreg_t *a, int id, void* d_buffer_ptr)
{
	int i, n_pri;
	int_v z = {0,0,0};
	if (n == 0) return 0;
	for (i = n_pri = 0; i < n; ++i) {
		a[i].sub = a[i].alt_sc = 0, a[i].secondary = a[i].secondary_all = -1, a[i].hash = hash_64(id+i);
		if (!a[i].is_alt) ++n_pri;
	}
	ks_introsort_mem_ars_hash(n, a, d_buffer_ptr);
	mem_mark_primary_se_core_GPU(d_opt, n, a, &z, d_buffer_ptr);
	for (i = 0; i < n; ++i) {
		mem_alnreg_t *p = &a[i];
		p->secondary_all = i; // keep the rank in the first round
		if (!p->is_alt && p->secondary >= 0 && a[p->secondary].is_alt)
			p->alt_sc = a[p->secondary].score;
	}
	if (n_pri >= 0 && n_pri < n) {
		// kv_resize(int, z, n);
		z.m = n; z.a = (int*)CUDAKernelRealloc(d_buffer_ptr, z.a, sizeof(int) * z.m, 4);
		if (n_pri > 0) ks_introsort_mem_ars_hash2(n, a, d_buffer_ptr);
		for (i = 0; i < n; ++i) z.a[a[i].secondary_all] = i;
		for (i = 0; i < n; ++i) {
			if (a[i].secondary >= 0) {
				a[i].secondary_all = z.a[a[i].secondary];
				if (a[i].is_alt) a[i].secondary = INT_MAX;
			} else a[i].secondary_all = -1;
		}
		if (n_pri > 0) { // mark primary for hits to the primary assembly only
			for (i = 0; i < n_pri; ++i) a[i].sub = 0, a[i].secondary = -1;
			mem_mark_primary_se_core_GPU(d_opt, n_pri, a, &z, d_buffer_ptr);
		}
	} else {
		for (i = 0; i < n; ++i)
			a[i].secondary_all = a[i].secondary;
	}
	// free(z.a);
	return n_pri;
}


__device__ static void mem_reorder_primary5(int T, mem_alnreg_v *a)
{
	int k, n_pri = 0, left_st = INT_MAX, left_k = -1;
	mem_alnreg_t t;
	for (k = 0; k < a->n; ++k)
		if (a->a[k].secondary < 0 && !a->a[k].is_alt && a->a[k].score >= T) ++n_pri;
	if (n_pri <= 1) return; // only one alignment
	for (k = 0; k < a->n; ++k) {
		mem_alnreg_t *p = &a->a[k];
		if (p->secondary >= 0 || p->is_alt || p->score < T) continue;
		if (p->qb < left_st) left_st = p->qb, left_k = k;
	}
	// assert(a->a[0].secondary < 0);
	if (left_k == 0) return; // no need to reorder
	t = a->a[0], a->a[0] = a->a[left_k], a->a[left_k] = t;
	for (k = 1; k < a->n; ++k) { // update secondary and secondary_all
		mem_alnreg_t *p = &a->a[k];
		if (p->secondary == 0) p->secondary = left_k;
		else if (p->secondary == left_k) p->secondary = 0;
		if (p->secondary_all == 0) p->secondary_all = left_k;
		else if (p->secondary_all == left_k) p->secondary_all = 0;
	}
}

// ONLY work after mem_mark_primary_se()
__device__ static char** mem_gen_alt(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, const mem_alnreg_v *a, int l_query, const char *query, void* d_buffer_ptr) 
{
	int i, k, r, *cnt, tot;
	kstring_t *aln = 0, str = {0,0,0};
	char **XA = 0, *has_alt;

	cnt = (int*)CUDAKernelCalloc(d_buffer_ptr, a->n, sizeof(int), 4);
	has_alt = (char*)CUDAKernelCalloc(d_buffer_ptr, a->n, 1, 1);
	for (i = 0, tot = 0; i < a->n; ++i) {
		// r = get_pri_idx(opt->XA_drop_ratio, a->a, i);
		int kk = a->a[i].secondary_all;
		if (kk >= 0 && a->a[i].score >= a->a[kk].score * opt->XA_drop_ratio) r = kk;
		else r = -1;
		if (r >= 0) {
			++cnt[r], ++tot;
			if (a->a[i].is_alt) has_alt[r] = 1;
		}
	}

	if (tot == 0) goto end_gen_alt;
	aln =(kstring_t*)CUDAKernelCalloc(d_buffer_ptr, a->n, sizeof(kstring_t), 8);
	for (i = 0; i < a->n; ++i) {
		mem_aln_t t;
		// if ((r = get_pri_idx(opt->XA_drop_ratio, a->a, i)) < 0) continue;
		int kk = a->a[i].secondary_all;
		if (kk >= 0 && a->a[i].score >= a->a[kk].score * opt->XA_drop_ratio) r = kk;
		else r = -1;
		if (r<0) continue;
		if (cnt[r] > opt->max_XA_hits_alt || (!has_alt[r] && cnt[r] > opt->max_XA_hits)) continue;
		t = mem_reg2aln_GPU(opt, bns, pac, l_query, query, &a->a[i], d_buffer_ptr);
		str.l = 0;
		kputs(bns->anns[t.rid].name, &str, d_buffer_ptr);
		kputc(',', &str, d_buffer_ptr); kputc("+-"[t.is_rev], &str, d_buffer_ptr); kputl(t.pos + 1, &str, d_buffer_ptr);
		kputc(',', &str, d_buffer_ptr);
		for (k = 0; k < t.n_cigar; ++k) {
			kputw(t.cigar[k]>>4, &str, d_buffer_ptr);
			kputc("MIDSHN"[t.cigar[k]&0xf], &str, d_buffer_ptr);
		}
		kputc(',', &str, d_buffer_ptr); kputw(t.NM, &str, d_buffer_ptr);
		if (opt->flag & MEM_F_XB) {
			kputc(',', &str, d_buffer_ptr);
			kputw(t.score, &str, d_buffer_ptr);
		}
		kputc(';', &str, d_buffer_ptr);
// 		free(t.cigar);
		kputsn(str.s, str.l, &aln[r], d_buffer_ptr);
	}
	XA = (char**)CUDAKernelCalloc(d_buffer_ptr, a->n, sizeof(char*), 8);
	for (k = 0; k < a->n; ++k)
		XA[k] = aln[k].s;

end_gen_alt:
// 	free(has_alt); free(cnt); free(aln); free(str.s);
	return XA;
}

extern __device__ char* d_seq_sam_ptr;
extern __device__ int d_seq_sam_offset;
__device__ static void mem_reg2sam(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, bseq1_t *s, mem_alnreg_v *a, int extra_flag, const mem_aln_t *m, void* d_buffer_ptr)
{
	kstring_t str;
	struct { size_t n, m; mem_aln_t *a; } aa;
	int k, l;
	char **XA = 0;

	if (!(opt->flag & MEM_F_ALL))
		XA = mem_gen_alt(opt, bns, pac, a, s->l_seq, s->seq, d_buffer_ptr);
	aa.n = 0; aa.m = 0; aa.a = 0;
	str.l = str.m = 0; str.s = 0;
	for (k = l = 0; k < a->n; ++k) {
		mem_alnreg_t *p = &a->a[k];
		mem_aln_t *q;
		if (p->score < opt->T) continue;
		if (p->secondary >= 0 && (p->is_alt || !(opt->flag&MEM_F_ALL))) continue;
		if (p->secondary >= 0 && p->secondary < INT_MAX && p->score < a->a[p->secondary].score * opt->drop_ratio) continue;
		// q = kv_pushp(mem_aln_t, aa);
		q = (((aa.n == aa.m)
			?(aa.m = (aa.m? aa.m<<1 : 2),
				aa.a = (mem_aln_t*)CUDAKernelRealloc(d_buffer_ptr, aa.a, sizeof(mem_aln_t) * aa.m, 8), 0)
			: 0), &aa.a[aa.n++]);

		*q = mem_reg2aln_GPU(opt, bns, pac, s->l_seq, s->seq, p, d_buffer_ptr);
		// assert(q->rid >= 0); // this should not happen with the new code
		q->XA = XA? XA[k] : 0;
		q->flag |= extra_flag; // flag secondary
		if (p->secondary >= 0) q->sub = -1; // don't output sub-optimal score
		if (l && p->secondary < 0) // if supplementary
			q->flag |= (opt->flag&MEM_F_NO_MULTI)? 0x10000 : 0x800;
		if (!(opt->flag & MEM_F_KEEP_SUPP_MAPQ) && l && !p->is_alt && q->mapq > aa.a[0].mapq)
			q->mapq = aa.a[0].mapq; // lower mapq for supplementary mappings, unless -5 or -q is applied
		++l;
	}
	if (aa.n == 0) { // no alignments good enough; then write an unaligned record
		mem_aln_t t;
		t = mem_reg2aln_GPU(opt, bns, pac, s->l_seq, s->seq, 0, d_buffer_ptr);
		t.flag |= extra_flag;
		mem_aln2sam(opt, bns, &str, s, 1, &t, 0, m, d_buffer_ptr);
	} else {
		for (k = 0; k < aa.n; ++k)
			mem_aln2sam(opt, bns, &str, s, aa.n, aa.a, k, m, d_buffer_ptr);
		// for (k = 0; k < aa.n; ++k) free(aa.a[k].cigar);
		// free(aa.a);
	}
	l = strlen_GPU(str.s); 		// length of output
	k = atomicAdd(&d_seq_sam_offset, l+1);	// offset to output to d_seq_sam_ptr
	memcpy(&d_seq_sam_ptr[k], str.s, l+1);	// copy sam to output
	s->sam = (char*)k; 	// record offset
	// if (XA) {
	// 	for (k = 0; k < a->n; ++k) free(XA[k]);
	// 	free(XA);
	// }
}

/**********************************************
 * Device functions for paired-end alignments *
 **********************************************/
#define raw_mapq(diff, a) ((int)(6.02 * (diff) / (a) + .499))

__device__ static int mem_pair(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, const mem_pestat_t pes[4], bseq1_t s[2], mem_alnreg_v a[2], int id, int *sub, int *n_sub, int z[2], int n_pri[2], void* d_buffer_ptr)
{
	pair64_v v, u;
	int r, i, k, y[4], ret; // y[] keeps the last hit
	int64_t l_pac = bns->l_pac;
	v.n = 0; v.m = 0; v.a = 0;
	u.n = 0; u.m = 0; u.a = 0;
	for (r = 0; r < 2; ++r) { // loop through read number
		for (i = 0; i < n_pri[r]; ++i) {
			pair64_t key;
			mem_alnreg_t *e = &a[r].a[i];
			key.x = e->rb < l_pac? e->rb : (l_pac<<1) - 1 - e->rb; // forward position
			key.x = (uint64_t)e->rid<<32 | (key.x - bns->anns[e->rid].offset);
			key.y = (uint64_t)e->score << 32 | i << 2 | (e->rb >= l_pac)<<1 | r;
			// kv_push(pair64_t, v=v, x=key);
			if (v.n == v.m) {
				v.m = (v).m? v.m<<1 : 2;
				v.a = (pair64_t*)CUDAKernelRealloc(d_buffer_ptr, v.a, sizeof(pair64_t) * v.m, 8);
			}
			v.a[v.n++] = key;
		}
	}
	ks_introsort_128(v.n, v.a, d_buffer_ptr);
	y[0] = y[1] = y[2] = y[3] = -1;
	//for (i = 0; i < v.n; ++i) printf("[%d]\t%d\t%c%ld\n", i, (int)(v.a[i].y&1)+1, "+-"[v.a[i].y>>1&1], (long)v.a[i].x);
	for (i = 0; i < v.n; ++i) {
		for (r = 0; r < 2; ++r) { // loop through direction
			int dir = r<<1 | (v.a[i].y>>1&1), which;
			if (pes[dir].failed) continue; // invalid orientation
			which = r<<1 | ((v.a[i].y&1)^1);
			if (y[which] < 0) continue; // no previous hits
			for (k = y[which]; k >= 0; --k) { // TODO: this is a O(n^2) solution in the worst case; remember to check if this loop takes a lot of time (I doubt)
				int64_t dist;
				int q;
				double ns;
				pair64_t *p;
				if ((v.a[k].y&3) != which) continue;
				dist = (int64_t)v.a[i].x - v.a[k].x;
				//printf("%d: %lld\n", k, dist);
				if (dist > pes[dir].high) break;
				if (dist < pes[dir].low)  continue;
				ns = (dist - pes[dir].avg) / pes[dir].std;
				q = (int)((v.a[i].y>>32) + (v.a[k].y>>32) + .721 * log(2. * erfc(fabs(ns) * M_SQRT1_2)) * opt->a + .499); // .721 = 1/log(4)
				if (q < 0) q = 0;
				// p = kv_pushp(pair64_t, u);
				p = (((u.n == u.m)
				   	? (u.m = (u.m? u.m<<1 : 2), u.a = (pair64_t*)CUDAKernelRealloc(d_buffer_ptr, u.a, sizeof(pair64_t) * u.m, 8), 0)
					: 0), &u.a[u.n++]);
				p->y = (uint64_t)k<<32 | i;
				p->x = (uint64_t)q<<32 | (hash_64(p->y ^ id<<8) & 0xffffffffU);
				//printf("[%lld,%lld]\t%d\tdist=%ld\n", v.a[k].x, v.a[i].x, q, (long)dist);
			}
		}
		y[v.a[i].y&3] = i;
	}
	if (u.n) { // found at least one proper pair
		int tmp = opt->a + opt->b;
		tmp = tmp > opt->o_del + opt->e_del? tmp : opt->o_del + opt->e_del;
		tmp = tmp > opt->o_ins + opt->e_ins? tmp : opt->o_ins + opt->e_ins;
		ks_introsort_128(u.n, u.a, d_buffer_ptr);
		i = u.a[u.n-1].y >> 32; k = u.a[u.n-1].y << 32 >> 32;
		z[v.a[i].y&1] = v.a[i].y<<32>>34; // index of the best pair
		z[v.a[k].y&1] = v.a[k].y<<32>>34;
		ret = u.a[u.n-1].x >> 32;
		*sub = u.n > 1? u.a[u.n-2].x>>32 : 0;
		for (i = (long)u.n - 2, *n_sub = 0; i >= 0; --i)
			if (*sub - (int)(u.a[i].x>>32) <= tmp) ++*n_sub;
	} else ret = 0, *sub = 0, *n_sub = 0;
	// free(u.a); free(v.a);
	return ret;
}

__device__ int mem_matesw(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, const mem_pestat_t pes[4], const mem_alnreg_t *a, int l_ms, const uint8_t *ms, mem_alnreg_v *ma, void* d_buffer_ptr)
{
	int64_t l_pac = bns->l_pac;
	int i, r, skip[4], n = 0, rid;
	for (r = 0; r < 4; ++r)
		skip[r] = pes[r].failed? 1 : 0;
	for (i = 0; i < ma->n; ++i) { // check which orinentation has been found
		int64_t dist;
		r = mem_infer_dir(l_pac, a->rb, ma->a[i].rb, &dist);
		if (dist >= pes[r].low && dist <= pes[r].high)
			skip[r] = 1;
	}
	if (skip[0] + skip[1] + skip[2] + skip[3] == 4) return 0; // consistent pair exist; no need to perform SW
	for (r = 0; r < 4; ++r) {
		int is_rev, is_larger;
		uint8_t *seq, *rev = 0, *ref = 0;
		int64_t rb, re;
		if (skip[r]) continue;
		is_rev = (r>>1 != (r&1)); // whether to reverse complement the mate
		is_larger = !(r>>1); // whether the mate has larger coordinate
		if (is_rev) {
			rev = (uint8_t*)CUDAKernelMalloc(d_buffer_ptr, l_ms, 1); // this is the reverse complement of $ms
			for (i = 0; i < l_ms; ++i) rev[l_ms - 1 - i] = ms[i] < 4? 3 - ms[i] : 4;
			seq = rev;
		} else seq = (uint8_t*)ms;
		if (!is_rev) {
			rb = is_larger? a->rb + pes[r].low : a->rb - pes[r].high;
			re = (is_larger? a->rb + pes[r].high: a->rb - pes[r].low) + l_ms; // if on the same strand, end position should be larger to make room for the seq length
		} else {
			rb = (is_larger? a->rb + pes[r].low : a->rb - pes[r].high) - l_ms; // similarly on opposite strands
			re = is_larger? a->rb + pes[r].high: a->rb - pes[r].low;
		}
		if (rb < 0) rb = 0;
		if (re > l_pac<<1) re = l_pac<<1;
		if (rb < re) ref = bns_fetch_seq_gpu(bns, pac, &rb, (rb+re)>>1, &re, &rid, d_buffer_ptr);
		if (a->rid == rid && re - rb >= opt->min_seed_len) { // no funny things happening
			kswr_t aln;
			mem_alnreg_t b;
			int tmp, xtra = KSW_XSUBO | KSW_XSTART | (l_ms * opt->a < 250? KSW_XBYTE : 0) | (opt->min_seed_len * opt->a);
			aln = ksw_align2(l_ms, seq, re - rb, ref, 5, opt->mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, xtra, 0, d_buffer_ptr);
			memset(&b, 0, sizeof(mem_alnreg_t));
			if (aln.score >= opt->min_seed_len && aln.qb >= 0) { // something goes wrong if aln.qb < 0
				b.rid = a->rid;
				b.is_alt = a->is_alt;
				b.qb = is_rev? l_ms - (aln.qe + 1) : aln.qb;                                                                                                                                                                              
				b.qe = is_rev? l_ms - aln.qb : aln.qe + 1; 
				b.rb = is_rev? (l_pac<<1) - (rb + aln.te + 1) : rb + aln.tb;
				b.re = is_rev? (l_pac<<1) - (rb + aln.tb) : rb + aln.te + 1;
				b.score = aln.score;
				b.csub = aln.score2;
				b.secondary = -1;
				b.seedcov = (b.re - b.rb < b.qe - b.qb? b.re - b.rb : b.qe - b.qb) >> 1;
//				printf("*** %d, [%lld,%lld], %d:%d, (%lld,%lld), (%lld,%lld) == (%lld,%lld)\n", aln.score, rb, re, is_rev, is_larger, a->rb, a->re, ma->a[0].rb, ma->a[0].re, b.rb, b.re);
				// kv_push(mem_alnreg_t, v=*ma, x=b); // make room for a new element
				if (ma->n == ma->m) {
					ma->m = ma->m? ma->m<<1 : 2;
					ma->a = (mem_alnreg_t*)CUDAKernelRealloc(d_buffer_ptr, ma->a, sizeof(mem_alnreg_t) * ma->m, 8);
				}
				ma->a[ma->n++] = b;
				// move b s.t. ma is sorted
				for (i = 0; i < ma->n - 1; ++i) // find the insertion point
					if (ma->a[i].score < b.score) break;
				tmp = i;
				for (i = ma->n - 1; i > tmp; --i) ma->a[i] = ma->a[i-1];
				ma->a[i] = b;
			}
			++n;
		}
		if (n) ma->n = mem_sort_dedup_patch(opt, 0, 0, 0, ma->n, ma->a, d_buffer_ptr);
		// if (rev) free(rev);
		// free(ref);
	}
	return n;
}


__device__ static int mem_sam_pe(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, const mem_pestat_t pes[4], uint64_t id, bseq1_t s[2], mem_alnreg_v a[2], void* d_buffer_ptr)
{
	int n = 0, i, j, z[2], o, subo, n_sub, extra_flag = 1, n_pri[2], n_aa[2];
	kstring_t str;
	mem_aln_t h[2], g[2], aa[2][2];

	str.l = str.m = 0; str.s = 0;
	memset(h, 0, sizeof(mem_aln_t) * 2);
	memset(g, 0, sizeof(mem_aln_t) * 2);
	n_aa[0] = n_aa[1] = 0;
	if (!(opt->flag & MEM_F_NO_RESCUE)) { // then perform SW for the best alignment
		mem_alnreg_v b[2];
		b[0].n = 0; b[0].m = 0; b[0].a = 0;
		b[1].n = 0; b[1].m = 0; b[1].a = 0;
		for (i = 0; i < 2; ++i)
			for (j = 0; j < a[i].n; ++j)
				if (a[i].a[j].score >= a[i].a[0].score  - opt->pen_unpaired){
					// kv_push(mem_alnreg_t, v=b[i], x=a[i].a[j]);
					if (b[i].n == b[i].m) {
						b[i].m = b[i].m? b[i].m<<1 : 2;
						b[i].a = (mem_alnreg_t*)CUDAKernelRealloc(d_buffer_ptr, b[i].a, sizeof(mem_alnreg_t) * b[i].m, 8);
					}
					b[i].a[b[i].n++] = a[i].a[j];
				}
		for (i = 0; i < 2; ++i)
			for (j = 0; j < b[i].n && j < opt->max_matesw; ++j)
				n += mem_matesw(opt, bns, pac, pes, &b[i].a[j], s[!i].l_seq, (uint8_t*)s[!i].seq, &a[!i], d_buffer_ptr);
		// free(b[0].a); free(b[1].a);
	}
	n_pri[0] = mem_mark_primary_se_GPU(opt, a[0].n, a[0].a, id<<1|0, d_buffer_ptr);
	n_pri[1] = mem_mark_primary_se_GPU(opt, a[1].n, a[1].a, id<<1|1, d_buffer_ptr);
	if (opt->flag & MEM_F_PRIMARY5) {
		mem_reorder_primary5(opt->T, &a[0]);
		mem_reorder_primary5(opt->T, &a[1]);
	}
	if (opt->flag&MEM_F_NOPAIRING) goto no_pairing;
	// pairing single-end hits
	if (n_pri[0] && n_pri[1] && (o = mem_pair(opt, bns, pac, pes, s, a, id, &subo, &n_sub, z, n_pri, d_buffer_ptr)) > 0) {
		int is_multi[2], q_pe, score_un, q_se[2];
		char **XA[2];
		// check if an end has multiple hits even after mate-SW
		for (i = 0; i < 2; ++i) {
			for (j = 1; j < n_pri[i]; ++j)
				if (a[i].a[j].secondary < 0 && a[i].a[j].score >= opt->T) break;
			is_multi[i] = j < n_pri[i]? 1 : 0;
		}
		if (is_multi[0] || is_multi[1]) goto no_pairing; // TODO: in rare cases, the true hit may be long but with low score
		// compute mapQ for the best SE hit
		score_un = a[0].a[0].score + a[1].a[0].score - opt->pen_unpaired;
		//q_pe = o && subo < o? (int)(MEM_MAPQ_COEF * (1. - (double)subo / o) * log(a[0].a[z[0]].seedcov + a[1].a[z[1]].seedcov) + .499) : 0;
		subo = subo > score_un? subo : score_un;
		q_pe = raw_mapq(o - subo, opt->a);
		if (n_sub > 0) q_pe -= (int)(4.343 * log((double)n_sub+1) + .499);
		if (q_pe < 0) q_pe = 0;
		if (q_pe > 60) q_pe = 60;
		q_pe = (int)(q_pe * (1. - .5 * (a[0].a[0].frac_rep + a[1].a[0].frac_rep)) + .499);
		// the following assumes no split hits
		if (o > score_un) { // paired alignment is preferred
			mem_alnreg_t *c[2];
			c[0] = &a[0].a[z[0]]; c[1] = &a[1].a[z[1]];
			for (i = 0; i < 2; ++i) {
				if (c[i]->secondary >= 0)
					c[i]->sub = a[i].a[c[i]->secondary].score, c[i]->secondary = -2;
				q_se[i] = mem_approx_mapq_se(opt, c[i]);
			}
			q_se[0] = q_se[0] > q_pe? q_se[0] : q_pe < q_se[0] + 40? q_pe : q_se[0] + 40;
			q_se[1] = q_se[1] > q_pe? q_se[1] : q_pe < q_se[1] + 40? q_pe : q_se[1] + 40;
			extra_flag |= 2;
			// cap at the tandem repeat score
			q_se[0] = q_se[0] < raw_mapq(c[0]->score - c[0]->csub, opt->a)? q_se[0] : raw_mapq(c[0]->score - c[0]->csub, opt->a);
			q_se[1] = q_se[1] < raw_mapq(c[1]->score - c[1]->csub, opt->a)? q_se[1] : raw_mapq(c[1]->score - c[1]->csub, opt->a);
		} else { // the unpaired alignment is preferred
			z[0] = z[1] = 0;
			q_se[0] = mem_approx_mapq_se(opt, &a[0].a[0]);
			q_se[1] = mem_approx_mapq_se(opt, &a[1].a[0]);
		}
		for (i = 0; i < 2; ++i) {
			int k = a[i].a[z[i]].secondary_all;
			if (k >= 0 && k < n_pri[i]) { // switch secondary and primary if both of them are non-ALT
				// assert(a[i].a[k].secondary_all < 0);
				for (j = 0; j < a[i].n; ++j)
					if (a[i].a[j].secondary_all == k || j == k)
						a[i].a[j].secondary_all = z[i];
				a[i].a[z[i]].secondary_all = -1;
			}
		}
		if (!(opt->flag & MEM_F_ALL)) {
			for (i = 0; i < 2; ++i)
				XA[i] = mem_gen_alt(opt, bns, pac, &a[i], s[i].l_seq, s[i].seq, d_buffer_ptr);
		} else XA[0] = XA[1] = 0;
		// write SAM
		for (i = 0; i < 2; ++i) {
			h[i] = mem_reg2aln_GPU(opt, bns, pac, s[i].l_seq, s[i].seq, &a[i].a[z[i]], d_buffer_ptr);
			h[i].mapq = q_se[i];
			h[i].flag |= 0x40<<i | extra_flag;
			h[i].XA = XA[i]? XA[i][z[i]] : 0;
			aa[i][n_aa[i]++] = h[i];
			if (n_pri[i] < a[i].n) { // the read has ALT hits
				mem_alnreg_t *p = &a[i].a[n_pri[i]];
				if (p->score < opt->T || p->secondary >= 0 || !p->is_alt) continue;
				g[i] = mem_reg2aln_GPU(opt, bns, pac, s[i].l_seq, s[i].seq, p, d_buffer_ptr);
				g[i].flag |= 0x800 | 0x40<<i | extra_flag;
				g[i].XA = XA[i]? XA[i][n_pri[i]] : 0;
				aa[i][n_aa[i]++] = g[i];
			}
		}
		for (i = 0; i < n_aa[0]; ++i)
			mem_aln2sam(opt, bns, &str, &s[0], n_aa[0], aa[0], i, &h[1], d_buffer_ptr); // write read1 hits
		int l_sam = strlen_GPU(str.s); 		// length of output
		int offset = atomicAdd(&d_seq_sam_offset, l_sam+1);	// offset to output to d_seq_sam_ptr
		memcpy(&d_seq_sam_ptr[offset], str.s, l_sam+1);	// copy sam to output
		s[0].sam = (char*)offset; 	// record offset
		str.l = 0;
		for (i = 0; i < n_aa[1]; ++i)
			mem_aln2sam(opt, bns, &str, &s[1], n_aa[1], aa[1], i, &h[0], d_buffer_ptr); // write read2 hits
		l_sam = strlen_GPU(str.s); 		// length of output
		offset = atomicAdd(&d_seq_sam_offset, l_sam+1);	// offset to output to d_seq_sam_ptr
		memcpy(&d_seq_sam_ptr[offset], str.s, l_sam+1);	// copy sam to output
		s[1].sam = (char*)offset; 	// record offset

		// if (strcmp(s[0].name, s[1].name) != 0) err_fatal(__func__, "paired reads have different names: \"%s\", \"%s\"\n", s[0].name, s[1].name);
// 		// free
// 		for (i = 0; i < 2; ++i) {
// 			free(h[i].cigar); free(g[i].cigar);
// 			if (XA[i] == 0) continue;
// 			for (j = 0; j < a[i].n; ++j) free(XA[i][j]);
// 			free(XA[i]);
// 		}
	} else goto no_pairing;
	return n;

no_pairing:
	for (i = 0; i < 2; ++i) {
		int which = -1;
		if (a[i].n) {
			if (a[i].a[0].score >= opt->T) which = 0;
			else if (n_pri[i] < a[i].n && a[i].a[n_pri[i]].score >= opt->T)
				which = n_pri[i];
		}
		if (which >= 0) h[i] = mem_reg2aln_GPU(opt, bns, pac, s[i].l_seq, s[i].seq, &a[i].a[which], d_buffer_ptr);
		else h[i] = mem_reg2aln_GPU(opt, bns, pac, s[i].l_seq, s[i].seq, 0, d_buffer_ptr);
	}
	if (!(opt->flag & MEM_F_NOPAIRING) && h[0].rid == h[1].rid && h[0].rid >= 0) { // if the top hits from the two ends constitute a proper pair, flag it.
		int64_t dist;
		int d;
		d = mem_infer_dir(bns->l_pac, a[0].a[0].rb, a[1].a[0].rb, &dist);
		if (!pes[d].failed && dist >= pes[d].low && dist <= pes[d].high) extra_flag |= 2;
	}
	mem_reg2sam(opt, bns, pac, &s[0], &a[0], 0x41|extra_flag, &h[1], d_buffer_ptr);
	mem_reg2sam(opt, bns, pac, &s[1], &a[1], 0x81|extra_flag, &h[0], d_buffer_ptr);
	// if (strcmp(s[0].name, s[1].name) != 0) err_fatal(__func__, "paired reads have different names: \"%s\", \"%s\"\n", s[0].name, s[1].name);
	// free(h[0].cigar); free(h[1].cigar);
	return n;
}



#define start_width 1
// first pass: find all SMEMs
__global__ void mem_collect_intv_kernel1(const mem_opt_t *opt, const bwt_t *bwt, const bseq1_t *d_seqs, 
	smem_aux_t *d_aux, 			// aux output
	int n,						// total number of reads
	void* d_buffer_pools)
{
	char *seq1; uint8_t *seq; int len;
	int i, x = 0;

	i = blockIdx.x*blockDim.x + threadIdx.x;		// ID of the read to process
	if (i>=n) return;
	void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, threadIdx.x % 32);	// set buffer pool
	smem_aux_t* a = &d_aux[i];						// get the aux for this read and init aux members
	a->tmpv[0] = (bwtintv_v*)CUDAKernelCalloc(d_buffer_ptr, 1, sizeof(bwtintv_v), 8);
	a->tmpv[0]->m = 30; a->tmpv[0]->a = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr, 30*sizeof(bwtintv_t), 8);
	a->tmpv[1] = (bwtintv_v*)CUDAKernelCalloc(d_buffer_ptr, 1, sizeof(bwtintv_v), 8);
	a->tmpv[1]->m = 30; a->tmpv[1]->a = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr, 30*sizeof(bwtintv_t), 8);
	a->mem.m   = 30; a->mem.a  = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr, 30*sizeof(bwtintv_t), 8);
	a->mem1.m  = 30; a->mem1.a = (bwtintv_t*)CUDAKernelMalloc(d_buffer_ptr, 30*sizeof(bwtintv_t), 8);
	seq1 = d_seqs[i].seq; 							// get seq from global mem
	len  = d_seqs[i].l_seq;
	if (len < opt->min_seed_len) return; 			// if the query is shorter than the seed length, no match

	// convert to 2-bit encoding if we have not done so
	for (i = 0; i < len; ++i)
		seq1[i] = seq1[i] < 4? seq1[i] : d_nst_nt4_table[(int)seq1[i]];
	seq = (uint8_t*)seq1;

	// first pass: find all SMEMs
	while (x < len) {
		if (seq[x] < 4) {
			x = bwt_smem1a_gpu(bwt, len, seq, x, start_width, 0, &a->mem1, a->tmpv, d_buffer_ptr);
			for (i = 0; i < a->mem1.n; ++i) {
				bwtintv_t *p = &a->mem1.a[i];
				int slen = (uint32_t)p->info - (p->info>>32); // seed length
				if (slen >= opt->min_seed_len){
					// kv_push(bwtintv_t, v=a->mem, x=*p, d_buffer_ptr);
					if (a->mem.n == a->mem.m) {
						a->mem.m = a->mem.m? a->mem.m<<1 : 2;
						a->mem.a = (bwtintv_t*)CUDAKernelRealloc(d_buffer_ptr, a->mem.a, sizeof(bwtintv_t) * a->mem.m, 8);
					}
					a->mem.a[a->mem.n++] = *p;
				}
			}
		} else ++x;
	}
}

// second pass: find MEMs inside a long SMEM
__global__ void mem_collect_intv_kernel2(const mem_opt_t *opt, const bwt_t *bwt, const bseq1_t *d_seqs, 
	smem_aux_t *d_aux, 			// aux output
	int n,						// total number of reads
	void* d_buffer_pools)
{
	uint8_t *seq; int len;
	int i, k;

	i = blockIdx.x*blockDim.x + threadIdx.x;		// ID of the read to process
	if (i>=n) return;
	void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, threadIdx.x % 32);	// set buffer pool
	seq = (uint8_t*)d_seqs[i].seq;
	len = d_seqs[i].l_seq;
	smem_aux_t* a = &d_aux[i];						// get the aux for this read 
	int old_n = a->mem.n;
	int split_len = (int)(opt->min_seed_len * opt->split_factor + .499);
	for (k = 0; k < old_n; ++k) {
		bwtintv_t *p = &a->mem.a[k];
		int start = p->info>>32, end = (int32_t)p->info;
		if (end - start < split_len || p->x[2] > opt->split_width) continue;
		bwt_smem1a_gpu(bwt, len, seq, (start + end)>>1, p->x[2]+1, 0, &a->mem1, a->tmpv, d_buffer_ptr);
		for (i = 0; i < a->mem1.n; ++i){
			if ((uint32_t)a->mem1.a[i].info - (a->mem1.a[i].info>>32) >= opt->min_seed_len){
				// kv_push(bwtintv_t, a->mem, a->mem1.a[i], d_buffer_ptr);
				if (a->mem.n == a->mem.m) {
					a->mem.m = a->mem.m? a->mem.m<<1 : 2;
					a->mem.a = (bwtintv_t*)CUDAKernelRealloc(d_buffer_ptr, a->mem.a, sizeof(bwtintv_t) * a->mem.m, 8);
				}
				a->mem.a[a->mem.n++] = a->mem1.a[i];
			}
		}
	}
}

// third pass: LAST-like
__global__ void mem_collect_intv_kernel3(const mem_opt_t *opt, const bwt_t *bwt, const bseq1_t *d_seqs, 
	smem_aux_t *d_aux, 			// aux output
	int n,						// total number of reads
	void* d_buffer_pools)
{
	uint8_t *seq; int len;
	int i;
	i = blockIdx.x*blockDim.x + threadIdx.x;		// ID of the read to process
	if (i>=n) return;
	void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, threadIdx.x % 32);	// set buffer pool
	seq = (uint8_t*)d_seqs[i].seq;
	len = d_seqs[i].l_seq;
	smem_aux_t* a = &d_aux[i];						// get the aux for this read and init aux members
	if (opt->max_mem_intv > 0) {
		int x = 0;
		while (x < len) {
			if (seq[x] < 4) {
				bwtintv_t m;
				x = bwt_seed_strategy1_gpu(bwt, len, seq, x, opt->min_seed_len, opt->max_mem_intv, &m);
				if (m.x[2] > 0) {
					// kv_push(bwtintv_t, a->mem, m, d_buffer_ptr);
					if (a->mem.n == a->mem.m) {
						a->mem.m = a->mem.m? a->mem.m<<1 : 2;
						a->mem.a = (bwtintv_t*)CUDAKernelRealloc(d_buffer_ptr, a->mem.a, sizeof(bwtintv_t) * a->mem.m, 8);
					}
					a->mem.a[a->mem.n++] = m;
				}
			} else ++x;
		}
	}
	// // sort
	ks_introsort(a->mem.n, a->mem.a, d_buffer_ptr);
}


__global__ void mem_chain_kernel(
	const mem_opt_t *opt,
	const bwt_t *bwt,
	const bntseq_t *bns,
	const bseq1_t *d_seqs,
	const int n,
	smem_aux_t *d_aux,
	mem_chain_v *d_chains,		// output
	void* d_buffer_pools
	)
{
	int i, b, e, l_rep;
	kbtree_chn_t *tree;
	mem_chain_v chain;

	i = blockIdx.x*blockDim.x + threadIdx.x;		// ID of the read to process
	if (i>=n) return;
	void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, threadIdx.x % 32);	// set buffer pool
	
	chain.n = 0; chain.m = 0, chain.a = 0;
	if (d_seqs[i].l_seq < opt->min_seed_len) { // if the query is shorter than the seed length, no match
		d_chains[i] = chain; return;
	}
	tree = kb_init_chn(512, d_buffer_ptr);
	smem_aux_t* aux = &d_aux[i];

	bwtintv_t p;
	for (i = 0, b = e = l_rep = 0; i < aux->mem.n; ++i) { // compute frac_rep
		p = aux->mem.a[i];
		int sb = (p.info>>32), se = (uint32_t)p.info;
		if (p.x[2] <= opt->max_occ) continue;
		if (sb > e) l_rep += e - b, b = sb, e = se;
		else e = e > se? e : se;
	}
	l_rep += e - b;

	for (i = 0; i < aux->mem.n; ++i) {
		p = aux->mem.a[i];
		int step, count, slen = (uint32_t)p.info - (p.info>>32); // seed length
		int64_t k;
		step = p.x[2] > opt->max_occ? p.x[2] / opt->max_occ : 1;
		for (k = count = 0; k < p.x[2] && count < opt->max_occ; k += step, ++count) {
			mem_chain_t tmp, *lower, *upper;
			mem_seed_t s;
			int rid, to_add = 0;
			s.rbeg = tmp.pos = bwt_sa_gpu(bwt, p.x[0] + k); // this is the base coordinate in the forward-reverse reference
			s.qbeg = p.info>>32;
			s.score= s.len = slen;
			rid = bns_intv2rid_gpu(bns, s.rbeg, s.rbeg + s.len);
			if (rid < 0) continue; // bridging multiple reference sequences or the forward-reverse boundary; TODO: split the seed; don't discard it!!!
			if (kb_size(tree)) {
				kb_intervalp_chn(tree, &tmp, &lower, &upper); // find the closest chain
				if (!lower || !test_and_merge(opt, bns->l_pac, lower, &s, rid, d_buffer_ptr)) to_add = 1;
			} else to_add = 1;
			if (to_add) { // add the seed as a new chain
				tmp.n = 1; tmp.m = 4;
				tmp.seeds = (mem_seed_t*)CUDAKernelCalloc(d_buffer_ptr, tmp.m, sizeof(mem_seed_t), 8);
				tmp.seeds[0] = s;
				tmp.rid = rid;
				tmp.is_alt = !!bns->anns[rid].is_alt;
				kb_putp_chn(tree, &tmp, d_buffer_ptr);
			}
		}
	}

	// // if (buf == 0) smem_aux_destroy(aux);

	// kv_resize(type = mem_chain_t, v = chain, s = kb_size(tree), d_buffer_ptr);
	chain.m = kb_size(tree);
	chain.a = (mem_chain_t*)CUDAKernelRealloc(d_buffer_ptr, chain.a, sizeof(mem_chain_t) * chain.m, 8);

	__kb_traverse(tree, &chain, d_buffer_ptr);
	b = d_seqs[blockIdx.x*blockDim.x + threadIdx.x].l_seq; // this is seq length
	for (i = 0; i < chain.n; ++i) chain.a[i].frac_rep = (float)l_rep / b;

// printf("unit test 2 chain n = %d\n", chain.n);
// printf("unit test 2 chain m = %d\n", chain.m);
// for (i=0; i<chain.n; i++) printf("unit test 2 chain rbeg = %ld\n", chain.a[i].seeds->rbeg);

	// kb_destroy(chn, tree);
	d_chains[blockIdx.x*blockDim.x + threadIdx.x] = chain;
}

__global__ void mem_chain_flt_kernel(const mem_opt_t *opt, 
	mem_chain_v *d_chains, 	// input and output
	int n, // number of reads
	void* d_buffer_pools)
{
	int i, k, n_chn;
	mem_chain_t	*a;

	i = blockIdx.x*blockDim.x + threadIdx.x;		// ID of the read to process
	if (i>=n) return;
	void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, threadIdx.x % 32);	// set buffer pool	
	a = d_chains[i].a;
	n_chn = d_chains[i].n;

	struct { size_t n, m; int* a; } chains = {0,0,0}; // this keeps int indices of the non-overlapping chains
	if (n_chn == 0) return; // no need to filter
	// compute the weight of each chain and drop chains with small weight
	for (i = k = 0; i < n_chn; ++i) {
		mem_chain_t *c = &a[i];
		c->first = -1; c->kept = 0;
		c->w = mem_chain_weight(c);
		if (c->w < opt->min_chain_weight) {} // free(c->seeds);
		else a[k++] = *c;
	}
	n_chn = k;
	ks_introsort_mem_flt(n_chn, a, d_buffer_ptr);
	// pairwise chain comparisons
	a[0].kept = 3;
	// kv_push(type=int, v=chains, x=0);
	if (chains.n == chains.m) {
		chains.m = chains.m? chains.m<<1 : 2;
		chains.a = (int*)CUDAKernelRealloc(d_buffer_ptr, chains.a, sizeof(int) * chains.m, 4);
	}
	chains.a[chains.n++] = 0;
	for (i = 1; i < n_chn; ++i) {
		int large_ovlp = 0;
		for (k = 0; k < chains.n; ++k) {
			int j = chains.a[k];
			int b_max = chn_beg(a[j]) > chn_beg(a[i])? chn_beg(a[j]) : chn_beg(a[i]);
			int e_min = chn_end(a[j]) < chn_end(a[i])? chn_end(a[j]) : chn_end(a[i]);
			if (e_min > b_max && (!a[j].is_alt || a[i].is_alt)) { // have overlap; don't consider ovlp where the kept chain is ALT while the current chain is primary
				int li = chn_end(a[i]) - chn_beg(a[i]);
				int lj = chn_end(a[j]) - chn_beg(a[j]);
				int min_l = li < lj? li : lj;
				if (e_min - b_max >= min_l * opt->mask_level && min_l < opt->max_chain_gap) { // significant overlap
					large_ovlp = 1;
					if (a[j].first < 0) a[j].first = i; // keep the first shadowed hit s.t. mapq can be more accurate
					if (a[i].w < a[j].w * opt->drop_ratio && a[j].w - a[i].w >= opt->min_seed_len<<1)
						break;
				}
			}
		}
		if (k == chains.n) {
	 		// kv_push(int, chains, i);
	 		if (chains.n == chains.m) {
				chains.m = chains.m? chains.m<<1 : 2;
				chains.a = (int*)CUDAKernelRealloc(d_buffer_ptr, chains.a, sizeof(int) * chains.m, 4);
			}
			chains.a[chains.n++] = i;
			a[i].kept = large_ovlp? 2 : 3;
		}
	}
	for (i = 0; i < chains.n; ++i) {
		mem_chain_t *c = &a[chains.a[i]];
		if (c->first >= 0) a[c->first].kept = 1;
	}
	// free(chains.a);
	for (i = k = 0; i < n_chn; ++i) { // don't extend more than opt->max_chain_extend .kept=1/2 chains
		if (a[i].kept == 0 || a[i].kept == 3) continue;
		if (++k >= opt->max_chain_extend) break;
	}
	for (; i < n_chn; ++i)
		if (a[i].kept < 3) a[i].kept = 0;
	for (i = k = 0; i < n_chn; ++i) { // free discarded chains
		mem_chain_t *c = &a[i];
		if (c->kept == 0){} // free(c->seeds);
		else a[k++] = a[i];
	}
	d_chains[blockIdx.x*blockDim.x + threadIdx.x].n = k;
}

__global__ void mem_flt_chained_seeds_kernel(
	const mem_opt_t *d_opt, const bntseq_t *d_bns, const uint8_t *d_pac, const bseq1_t *d_seqs,
	mem_chain_v *d_chains, 	// input and output
	int n,		// number of seqs
	void* d_buffer_pools
	)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;		// ID of the read to process
	if (i>=n) return;
	void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, threadIdx.x % 32);	// set buffer pool	
	mem_chain_t* a = d_chains[i].a;
	int n_chn = d_chains[i].n;
	uint8_t* query = (uint8_t*)d_seqs[i].seq;
	int l_query = d_seqs[i].l_seq;

	double min_l = d_opt->min_chain_weight? MEM_HSP_COEF * d_opt->min_chain_weight : MEM_MINSC_COEF * log((float)l_query);
	int j, k, min_HSP_score = (int)(d_opt->a * min_l + .499);
	if (min_l > MEM_SEEDSW_COEF * l_query) return; // don't run the following for short reads
	for (i = 0; i < n_chn; ++i) {
		mem_chain_t *c = &a[i];
		for (j = k = 0; j < c->n; ++j) {
			mem_seed_t *s = &c->seeds[j];
			s->score = mem_seed_sw(d_opt, d_bns, d_pac, l_query, query, s, d_buffer_ptr);
			if (s->score < 0 || s->score >= min_HSP_score) {
				s->score = s->score < 0? s->len * d_opt->a : s->score;
				c->seeds[k++] = *s;
			}
		}
		c->n = k;
	}
}

__global__ void mem_chain2aln_kernel(
	const mem_opt_t *d_opt,
	const bntseq_t *d_bns,
	const uint8_t *d_pac,
	int n, // number of reads
	const bseq1_t* d_seqs,
	mem_chain_v *d_chains, 	// input chains
	mem_alnreg_v* d_regs,		// output array
	void *d_buffer_pools
	)
{
	int j;
	j = blockIdx.x*blockDim.x + threadIdx.x;		// ID of the read to process
	if (j>=n) return;
	void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, threadIdx.x % 32);	// set buffer pool
	int l_seq = d_seqs[j].l_seq;
	uint8_t *seq = (uint8_t*)d_seqs[j].seq;
	mem_chain_v chn = d_chains[j];

	mem_alnreg_v regs;		// output regs
	regs.n = 0; regs.m = 0; regs.a = 0;

	for (j = 0; j < chn.n; ++j) {
		mem_chain_t *p = &chn.a[j];
		mem_chain2aln(d_opt, d_bns, d_pac, l_seq, (uint8_t*)seq, p, &regs, d_buffer_ptr);
	}
	// output
	d_regs[blockIdx.x*blockDim.x + threadIdx.x] = regs;
}

__global__ void mem_sort_dedup_patch_kernel(
	mem_opt_t *d_opt, 			// user-defined options
	bntseq_t *d_bns, 
	uint8_t *d_pac, 
	int n, 						// number of reads being processed in a batch
	bseq1_t* d_seqs,			// array of sequence info
	mem_alnreg_v* d_regs,		// array of output regs on GPU
	void* d_buffer_pools 		// for CUDA kernel memory management
	)
{
	int i;
	mem_alnreg_t *a;
	i = blockIdx.x*blockDim.x + threadIdx.x;		// ID of the read to process
	if (i>=n) return;
	void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, threadIdx.x % 32);	// set buffer pool
	n = d_regs[i].n;
	a = d_regs[i].a;
	uint8_t *seq = (uint8_t*)d_seqs[i].seq;

	n = mem_sort_dedup_patch(d_opt, d_bns, d_pac, (uint8_t*)seq, n, a, d_buffer_ptr);
	d_regs[i].n = n;
// printf("thread %d finished mem_sort_dedup_patch\n", i);
// printf("unit test 6 regs.n = %d\n", regs.n);	
	for (i = 0; i < n; ++i) {
		mem_alnreg_t *p = &a[i];
		if (p->rid >= 0 && d_bns->anns[p->rid].is_alt)
			p->is_alt = 1;
	}
}

/* ----------------------- MAIN FUNCTION FOR GENERATING ALIGNMENT RESULTS -----------------*/
__global__ void generate_sam_kernel(
	mem_opt_t *d_opt, 			// user-defined options
	bntseq_t *d_bns, 
	uint8_t *d_pac, 
	bseq1_t* d_seqs,			// array of sequence info
	int n, 						// number of reads being processed in a batch
	mem_pestat_t* d_pes,		// statistics for paired end
	mem_alnreg_v* d_regs,		// array of regs
	void* d_buffer_pools 		// for CUDA kernel memory management
	)
{
	void* d_buffer_ptr = CUDAKernelSelectPool(d_buffer_pools, threadIdx.x % 32); 	// the buffer pool this thread is using	
	uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;		// ID of the read to process

	if (i>=n) return; 	// don't run the padded threads

	if (!(d_opt->flag&MEM_F_PE)) {
		mem_mark_primary_se_GPU(d_opt, d_regs[i].n, d_regs[i].a, i, d_buffer_ptr);
		if (d_opt->flag & MEM_F_PRIMARY5) mem_reorder_primary5(d_opt->T, &d_regs[i]);
		mem_reg2sam(d_opt, d_bns, d_pac, &d_seqs[i], &d_regs[i], 0, 0, d_buffer_ptr);
		// free(w->regs[i].a);
	} else {
		mem_sam_pe(d_opt, d_bns, d_pac, d_pes, i, &d_seqs[i<<1], &d_regs[i<<1], d_buffer_ptr);
		// free(w->regs[i<<1|0].a); free(w->regs[i<<1|1].a);
	}
}

/*  main function for bwamem in GPU 
	return to seqs.sam
 */
void mem_align_GPU(gpu_ptrs_t gpu_data, bseq1_t* seqs, const mem_opt_t *opt, const bntseq_t *bns)
{
	/*GRID SIZE*/
	dim3 dimGrid(ceil((double)gpu_data.n_seqs/CUDA_BLOCKSIZE));
	dim3 dimBlock(CUDA_BLOCKSIZE);
	// debug
	// dim3 dimGrid(1);
	// dim3 dimBlock(32);


	/* first kernel: mem_collect_intv_kernel */
	// pre-allocate aux output
	smem_aux_t* d_aux;
	cudaMalloc((void**)&d_aux, gpu_data.n_seqs*sizeof(smem_aux_t));
	fprintf(stderr, "[M::%s] Launch kernel mem_collect_intv1 ...\n", __func__);
	mem_collect_intv_kernel1 <<< dimGrid, dimBlock, 0 >>> (
			gpu_data.d_opt, gpu_data.d_bwt, gpu_data.d_seqs, 
			d_aux,	// output
			gpu_data.n_seqs,
			gpu_data.d_buffer_pools);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	
	fprintf(stderr, "[M::%s] Launch kernel mem_collect_intv2 ...\n", __func__);
	mem_collect_intv_kernel2 <<< dimGrid, dimBlock, 0 >>> (
			gpu_data.d_opt, gpu_data.d_bwt, gpu_data.d_seqs, 
			d_aux,	// output
			gpu_data.n_seqs,
			gpu_data.d_buffer_pools);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	fprintf(stderr, "[M::%s] Launch kernel mem_collect_intv3 ...\n", __func__);
	mem_collect_intv_kernel3 <<< dimGrid, dimBlock, 0 >>> (
			gpu_data.d_opt, gpu_data.d_bwt, gpu_data.d_seqs, 
			d_aux,	// output
			gpu_data.n_seqs,
			gpu_data.d_buffer_pools);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	/* second kernel: chaining seeds */
	mem_chain_v *d_chains;
	cudaMalloc((void**)&d_chains, gpu_data.n_seqs*sizeof(mem_chain_v));
	fprintf(stderr, "[M::%s] Launch kernel mem_chain ...\n", __func__);
	mem_chain_kernel <<< dimGrid, dimBlock, 0 >>> (
			gpu_data.d_opt, gpu_data.d_bwt, gpu_data.d_bns, gpu_data.d_seqs,
			gpu_data.n_seqs,
			d_aux,
			d_chains,		// output
			gpu_data.d_buffer_pools);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	/* third kernel: chain filtering */
	fprintf(stderr, "[M::%s] Launch kernel mem_chain_flt ...\n", __func__);
	mem_chain_flt_kernel <<< dimGrid, dimBlock, 0 >>> (
			gpu_data.d_opt, 
			d_chains, 	// input and output
			gpu_data.n_seqs, 		// number of reads
			gpu_data.d_buffer_pools);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	/* fourth kernel: mem_flt_chained_seeds */
	fprintf(stderr, "[M::%s] Launch kernel mem_flt_chained_seeds ...\n", __func__);
	mem_flt_chained_seeds_kernel <<< dimGrid, dimBlock, 0 >>> (
			gpu_data.d_opt, 
			gpu_data.d_bns,
			gpu_data.d_pac,
			gpu_data.d_seqs,
			d_chains, 	// input and output
			gpu_data.n_seqs, 		// number of reads
			gpu_data.d_buffer_pools);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );


	/* fifth kernel: SW extension */
	mem_alnreg_v *d_regs;
	cudaMalloc((void**)&d_regs, gpu_data.n_seqs*sizeof(mem_alnreg_v));
	fprintf(stderr, "[M::%s] Launch kernel mem_chain2aln ...\n", __func__);
	mem_chain2aln_kernel <<< dimGrid, dimBlock, 0 >>> (
			gpu_data.d_opt,
			gpu_data.d_bns,
			gpu_data.d_pac,
			gpu_data.n_seqs, // number of reads
			gpu_data.d_seqs,
			d_chains, 		// input chains
			d_regs,			// output array
			gpu_data.d_buffer_pools);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	/* sixth kernel: mem_sort_dedup_patch */
	fprintf(stderr, "[M::%s] Launch kernel mem_sort_dedup_patch ...\n", __func__);
	mem_sort_dedup_patch_kernel <<< dimGrid, dimBlock, 0 >>> (
			gpu_data.d_opt,
			gpu_data.d_bns,
			gpu_data.d_pac,
			gpu_data.n_seqs,
			gpu_data.d_seqs,
			d_regs,
			gpu_data.d_buffer_pools
	);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );


	/* paired-end statistics */
	if (opt->flag&MEM_F_PE) { 
		// COPY d_regs to host memory 
		mem_alnreg_v* h_regs;
		h_regs = (mem_alnreg_v*)malloc(gpu_data.n_seqs * sizeof(mem_alnreg_v));
		cudaMemcpy(h_regs, d_regs, gpu_data.n_seqs*sizeof(mem_alnreg_v), cudaMemcpyDeviceToHost);
			// copy member array a
		mem_alnreg_t* temp_a;
		for (int i=0; i<gpu_data.n_seqs; i++ ){
			temp_a = (mem_alnreg_t*)malloc(h_regs[i].n*sizeof(mem_alnreg_t));
			cudaMemcpy(temp_a, h_regs[i].a, h_regs[i].n*sizeof(mem_alnreg_t), cudaMemcpyDeviceToHost);
			h_regs[i].a = temp_a;
		}
		// infer insert sizes if not provided
		mem_pestat_t pes[4];
		if (gpu_data.h_pes0) memcpy(pes, gpu_data.h_pes0, 4 * sizeof(mem_pestat_t)); 	// if pes0 != NULL, set the insert-size distribution as pes0
		else mem_pestat(opt, bns->l_pac, gpu_data.n_seqs, h_regs, pes); 		// otherwise, infer the insert size distribution from data
		// copy pes to device
		cudaMemcpy(gpu_data.d_pes, pes, 4*sizeof(mem_pestat_t), cudaMemcpyHostToDevice);
		// free intermediate data
		for (int i=0; i<gpu_data.n_seqs; i++ )
			free(h_regs[i].a);
		free(h_regs); 
	}
	
	/* generate SAM alignment */
	fprintf(stderr, "[M::%s] Launch kernel generate_sam ...\n", __func__);
	generate_sam_kernel <<< dimGrid, dimBlock >>> 
		(gpu_data.d_opt, gpu_data.d_bns, gpu_data.d_pac, 
		 gpu_data.d_seqs, gpu_data.n_seqs, gpu_data.d_pes, d_regs, gpu_data.d_buffer_pools);
	cudaPeekAtLastError() ;
	cudaDeviceSynchronize();
	fprintf(stderr, "[M::%s] Finished kernel generate_sam ...\n", __func__);

	/* copy sam output to host */
	bseq1_t* h_seqs;
	h_seqs = (bseq1_t*)malloc(gpu_data.n_seqs*sizeof(bseq1_t));
	cudaMemcpy(h_seqs, gpu_data.d_seqs, gpu_data.n_seqs*sizeof(bseq1_t), cudaMemcpyDeviceToHost);
	int L_sam;	// find total size of sam
	cudaMemcpyFromSymbol(&L_sam, d_seq_sam_offset, sizeof(int), 0, cudaMemcpyDeviceToHost);
	char* symbol_addr, *d_temp;
	gpuErrchk(cudaGetSymbolAddress((void**)&symbol_addr, d_seq_sam_ptr));
	gpuErrchk(cudaMemcpy(&d_temp, symbol_addr, sizeof(char*), cudaMemcpyDeviceToHost));
	cudaMemcpy(seq_sam_ptr, d_temp, L_sam, cudaMemcpyDeviceToHost);
	for (int i=0; i<gpu_data.n_seqs; i++)
		seqs[i].sam = &seq_sam_ptr[(long)h_seqs[i].sam];

	// free intermediate data
	cudaFree(d_aux); cudaFree(d_chains); cudaFree(d_regs);
	free(h_seqs);
};

/* Function to set up GPU memory
	copy constant data from host to device
	set up buffer pools
 */
gpu_ptrs_t GPU_Init(
	/* Input args */
	const mem_opt_t *opt, 
	const bwt_t *bwt, 
	const bntseq_t *bns, 
	const uint8_t *pac,
	mem_pestat_t *pes0
	)
{
	cudaSetDevice(0);
	// cudaDeviceSetLimit(cudaLimitStackSize, 2048);
	gpu_ptrs_t gpu_data;	// output
	CUDATransferStaticData(opt, bwt, bns, pac, pes0, &gpu_data);
	// buffer
	gpu_data.d_buffer_pools = CUDA_BufferInit();
	return gpu_data;
}

__device__ int d_seq_sam_offset = 0;
void prepare_batch_GPU(gpu_ptrs_t* gpu_data, const bseq1_t* seqs, int n_seqs, const mem_opt_t *opt){
	/* free d_seqs if they exist
	   reset buffer pools
	   update opt if opt !=0
	 */
	fprintf(stderr, "[M::%s] Prepare to align new batch of %d reads ..... \n", __func__, n_seqs);

	// transfer seqs
	CUDATransferSeqs(n_seqs);
	gpu_data->d_seqs = d_preallocated_seqs;
	gpu_data->n_seqs = n_seqs;

	// reset sam offset on device
	int zero = 0;
	// cudaGetSymbolAddress((void**)symbol_addr, "d_seq_sam_offset");
	gpuErrchk(cudaMemcpyToSymbol(d_seq_sam_offset, &zero, sizeof(int), 0, cudaMemcpyHostToDevice));

	// reset buffer pools
	CUDAResetBufferPool(gpu_data->d_buffer_pools);

	// update opt if opt != 0
	if (opt != 0){
		fprintf(stderr, "[M::%s] updating options ...\n", __func__);
		cudaMemcpy(gpu_data->d_opt, opt, sizeof(mem_opt_t), cudaMemcpyHostToDevice);
	}
}