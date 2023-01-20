/* The MIT License

   Copyright (c) 2018-     Dana-Farber Cancer Institute
                 2009-2018 Broad Institute, Inc.
                 2008-2009 Genome Research Ltd. (GRL)

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

#ifndef BWA_H_
#define BWA_H_

#include <stdint.h>
#include "bntseq.h"
#include "bwt.h"
#include "kmers_index/hashKMerIndex.h"

#define BWA_IDX_BWT 0x1
#define BWA_IDX_BNS 0x2
#define BWA_IDX_PAC 0x4
#define BWA_IDX_ALL 0x7

#define BWA_CTL_SIZE 0x10000

#define BWTALGO_AUTO  0
#define BWTALGO_RB2   1
#define BWTALGO_BWTSW 2
#define BWTALGO_IS    3

typedef struct {
	bwt_t    *bwt; // FM-index
	bntseq_t *bns; // information on the reference sequences
	uint8_t  *pac; // the actual 2-bit encoded reference sequences with 'N' converted to a random base

	int    is_shm;
	int64_t l_mem;
	uint8_t  *mem;
} bwaidx_t;

typedef struct {
	int l_seq, id;
	char *name, *comment, *seq, *qual, *sam;
	int8_t l_name, l_comment;
	int16_t l_qual;
} bseq1_t;

extern int bwa_verbose;
extern char bwa_rg_id[256];

#define MEM_MAPQ_COEF 30.0
#define MEM_MAPQ_MAX  60

struct __smem_i;
typedef struct __smem_i smem_i;

#define MEM_F_PE        0x2
#define MEM_F_NOPAIRING 0x4
#define MEM_F_ALL       0x8
#define MEM_F_NO_MULTI  0x10
#define MEM_F_NO_RESCUE 0x20
#define MEM_F_REF_HDR	0x100
#define MEM_F_SOFTCLIP  0x200
#define MEM_F_SMARTPE   0x400
#define MEM_F_PRIMARY5  0x800
#define MEM_F_KEEP_SUPP_MAPQ 0x1000
#define MEM_F_XB        0x2000

typedef struct {
	uint64_t max_mem_intv;
	int a, b;               // match score and mismatch penalty
	int o_del, e_del;
	int o_ins, e_ins;
	int pen_unpaired;       // phred-scaled penalty for unpaired reads
	int pen_clip5,pen_clip3;// clipping penalty. This score is not deducted from the DP score.
	int w;                  // band width
	int zdrop;              // Z-dropoff


	int T;                  // output score threshold; only affecting output
	int flag;               // see MEM_F_* macros
	int min_seed_len;       // minimum seed length
	int min_chain_weight;
	int max_chain_extend;
	float split_factor;     // split into a seed if MEM is longer than min_seed_len*split_factor
	int split_width;        // split into a seed if its occurence is smaller than this value
	int max_occ;            // skip a seed if its occurence is larger than this value
	int max_chain_gap;      // do not chain seed if it is max_chain_gap-bp away from the closest seed
	int n_threads;          // number of threads
	int chunk_size;         // process chunk_size-bp sequences in a batch
	float mask_level;       // regard a hit as redundant if the overlap with another better hit is over mask_level times the min length of the two hits
	float drop_ratio;       // drop a chain if its seed coverage is below drop_ratio times the seed coverage of a better chain overlapping with the small chain
	float XA_drop_ratio;    // when counting hits for the XA tag, ignore alignments with score < XA_drop_ratio * max_score; only effective for the XA tag
	float mask_level_redun;
	float mapQ_coef_len;
	int mapQ_coef_fac;
	int max_ins;            // when estimating insert size distribution, skip pairs with insert longer than this value
	int max_matesw;         // perform maximally max_matesw rounds of mate-SW for each end
	int max_XA_hits, max_XA_hits_alt; // if there are max_hits or fewer, output them all
	int8_t mat[25];         // scoring matrix; mat[0] == 0 if unset
} mem_opt_t;

// chaining data struct 
typedef struct {
	int64_t rbeg;
	int32_t qbeg, len;
	int score;
	float frac_rep;
	int rid;
	int CUDA_PADDING;
} mem_seed_t; // unaligned memory

typedef struct {
	int n;
	mem_seed_t *a;
} mem_seed_v;

typedef struct {
	mem_seed_t *seeds;
	int64_t pos;
	int16_t n, m, first, rid;
	uint32_t w:29, kept:2, is_alt:1;
	float frac_rep;
	// int64_t pos;
} mem_chain_t;
typedef struct { int n, m; mem_chain_t *a;  } mem_chain_v;

typedef struct {
	int64_t rb, re; // [rb,re): reference sequence in the alignment
	uint64_t hash;
	float frac_rep;
	int qb, qe;     // [qb,qe): query sequence in the alignment
	int rid;        // reference seq ID
	int score;      // best local SW score
	int truesc;     // actual score corresponding to the aligned region; possibly smaller than $score
	int sub;        // 2nd best SW score
	int alt_sc;
	int csub;       // SW score of a tandem hit
	int sub_n;      // approximate number of suboptimal hits
	int w;          // actual band width used in extension
	int seedcov;    // length of regions coverged by seeds
	int secondary;  // index of the parent hit shadowing the current hit; <0 if primary
	int secondary_all;
	int seedlen0;   // length of the starting seed
	int n_comp:30, is_alt:2; // number of sub-alignments chained together
} mem_alnreg_t;

typedef struct { int n, m; mem_alnreg_t *a; } mem_alnreg_v;

typedef struct {
	int low, high;   // lower and upper bounds within which a read pair is considered to be properly paired
	int failed;      // non-zero if the orientation is not supported by sufficient data
	double avg, std; // mean and stddev of the insert size distribution
} mem_pestat_t;

typedef struct { // This struct is only used for the convenience of API.
	int64_t pos;     // forward strand 5'-end mapping position
	char *XA;        // alternative mappings
	uint32_t *cigar; // CIGAR in the BAM encoding: opLen<<4|op; op to integer mapping: MIDSH=>01234
	int rid;         // reference sequence index in bntseq_t; <0 for unmapped
	int flag;        // extra flag
	uint32_t is_rev:1, is_alt:1, mapq:8, NM:22; // is_rev: whether on the reverse strand; mapq: mapping quality; NM: edit distance
	int n_cigar;     // number of CIGAR operations

	int score, sub, alt_sc;
} mem_aln_t;

typedef struct { int n; mem_aln_t *a; } mem_aln_v;

// temporary data processing on GPU
typedef struct
{
	int seqID;			// read ID
	uint16_t chainID;	// index on the chain vector of the read
	uint16_t seedID	;	// index of seed on the chain
	uint16_t regID;		// index on the (mem_alnreg_t)regs.a vector
	// below are for SW extension
	uint8_t* read_left; 	// string of read on the left of seed
	uint8_t* ref_left;		// string of reference on the left of seed
	uint8_t* read_right; 	// string of read on the right of seed
	uint8_t* ref_right;		// string of reference on the right of seed
	uint16_t readlen_left; 	// length of read on the left of seed
	uint16_t reflen_left;	// length of reference on the left of seed
	uint16_t readlen_right; // length of read on the right of seed
	uint16_t reflen_right;	// length of reference on the right of seed
} seed_record_t;

typedef struct {
	bwtintv_v mem, mem1, *tmpv[2];
} smem_aux_t;

typedef struct superbatch_data_t
{
	int n_reads;	   // number of reads
	bseq1_t *reads;	   // read info with pointers to the ones below
	char *name;		   // big chunk of all names
	char *comment;	   // big chunk of all comments
	char *seqs;		   // big chunk of all seqs
	char *qual;		   // big chunk of all qual
	long name_size;	   // total length of name strings
	long comment_size; // total length of comment strings
	long seqs_size;	   // total length of seq strings
	long qual_size;	   // total length of qual strings
} superbatch_data_t;

/* 
    data for processing  a GPU
 */

typedef struct {
    // constant pointers on device ( index, memory managment, user-options , etc. )
	mem_opt_t* d_opt;		// user-defined options
	bwt_t* d_bwt;			// bwt
	bntseq_t* d_bns;		
	uint8_t* d_pac; 
	void* d_buffer_pools;	// buffer pools
	mem_pestat_t* d_pes; 	// paired-end stats
	mem_pestat_t* h_pes0;	// pes0 on host for paired-end stats
	kmers_bucket_t* d_kmerHashTab;
    // pointers that will change each batch (being swapped between transfer and process)
        // reads on device
	int n_seqs;			// number of reads on device
	int64_t n_processed;	// number of reads processed prior to this batch
	bseq1_t *d_seqs;		// reads
    char *d_seq_name_ptr, *d_seq_comment_ptr, *d_seq_seq_ptr, *d_seq_qual_ptr, *d_seq_sam_ptr;  // name, comment, seq, qual, sam output
	int *d_seq_sam_size;	// length of sam on device, also used for all threads to atomic write SAM
	    // pre-allocated reads on host
    bseq1_t *h_seqs;		// reads
    char *h_seq_name_ptr, *h_seq_comment_ptr, *h_seq_seq_ptr, *h_seq_qual_ptr, *h_seq_sam_ptr;  // name, comment, seq, qual, sam output

        // intermediate data on device
	seed_record_t *d_seed_records; 	// global records of seeds, a big chunk of memory
	int *d_Nseeds;			// total number of seeds
	smem_aux_t* d_aux;		// collections of SA intervals, vector of size nseqs
	mem_seed_v* d_seq_seeds;// seeds array for each read
	mem_chain_v *d_chains;	// chain vectors of size nseqs
	mem_alnreg_v *d_regs;	// alignment info vectors, size nseqs
	mem_aln_v * d_alns;		// alignment vectors, size nseqs
	        // arrays for sorting, each has length = n_seqs
	int *d_sortkeys_in;
	int *d_seqIDs_in;
	int *d_sortkeys_out;
	int *d_seqIDs_out;
	int n_sortkeys;

    // pointers to CUDA stream, using generic pointers for compatibility with C
    void *CUDA_stream;   // process stream

} process_data_t;


/* 
    data for transferring from/to GPU
 */

typedef struct {
    // reads on device
	int n_seqs;			// number of reads
	int64_t total_input;	// number of reads input prior to this batch
	int64_t total_output;	// number of reads output prior to this batch
	bseq1_t *d_seqs;		// reads
    char *d_seq_name_ptr, *d_seq_comment_ptr, *d_seq_seq_ptr, *d_seq_qual_ptr, *d_seq_sam_ptr;  // name, comment, seq, qual, sam output
	int *d_seq_sam_size;	// length of sam on device
    // pre-allocated reads on host
    bseq1_t *h_seqs;		// reads
    char *h_seq_name_ptr, *h_seq_comment_ptr, *h_seq_seq_ptr, *h_seq_qual_ptr, *h_seq_sam_ptr;  // name, comment, seq, qual, sam output
	int h_seq_name_size, h_seq_comment_size, h_seq_seq_size, h_seq_qual_size; // total char length of name, comment, seq, qual in a batch
    // pointers to CUDA stream, using generic pointers for compatibility with C
    void *CUDA_stream;   // transfer stream
} transfer_data_t;




#ifdef __cplusplus
extern "C" {
#endif

	bseq1_t *bseq_read(int chunk_size, int *n_, void *ks1_, void *ks2_);
	void bseq_classify(int n, bseq1_t *seqs, int m[2], bseq1_t *sep[2]);

	// for cuda transfer and processing
	void bseq_read2(int chunk_size, int *n_, void *ks1_, void *ks2_, superbatch_data_t *transfer_data);

	void bwa_fill_scmat(int a, int b, int8_t mat[25]);
	uint32_t *bwa_gen_cigar(const int8_t mat[25], int q, int r, int w_, int64_t l_pac, const uint8_t *pac, int l_query, uint8_t *query, int64_t rb, int64_t re, int *score, int *n_cigar, int *NM);
	uint32_t *bwa_gen_cigar2(const int8_t mat[25], int o_del, int e_del, int o_ins, int e_ins, int w_, int64_t l_pac, const uint8_t *pac, int l_query, uint8_t *query, int64_t rb, int64_t re, int *score, int *n_cigar, int *NM);

	int bwa_idx_build(const char *fa, const char *prefix, int algo_type, int block_size);

	char *bwa_idx_infer_prefix(const char *hint);
	bwt_t *bwa_idx_load_bwt(const char *hint);

	bwaidx_t *bwa_idx_load_from_shm(const char *hint);
	bwaidx_t *bwa_idx_load_from_disk(const char *hint, int which);
	bwaidx_t *bwa_idx_load(const char *hint, int which);
	void bwa_idx_destroy(bwaidx_t *idx);
	int bwa_idx2mem(bwaidx_t *idx);
	int bwa_mem2idx(int64_t l_mem, uint8_t *mem, bwaidx_t *idx);

	void bwa_print_sam_hdr(const bntseq_t *bns, const char *hdr_line);
	char *bwa_set_rg(const char *s);
	char *bwa_insert_header(const char *s, char *hdr);

#ifdef __cplusplus
}
#endif

#endif
