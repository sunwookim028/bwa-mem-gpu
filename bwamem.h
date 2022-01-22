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
#ifndef BWAMEM_H_
#define BWAMEM_H_

#include "bwt.h"
#include "bntseq.h"
#include "bwa.h"

#ifdef __cplusplus
extern "C" {
#endif

	smem_i *smem_itr_init(const bwt_t *bwt);
	void smem_itr_destroy(smem_i *itr);
	void smem_set_query(smem_i *itr, int len, const uint8_t *query);
	void smem_config(smem_i *itr, int min_intv, int max_len, uint64_t max_intv);
	const bwtintv_v *smem_next(smem_i *itr);

	mem_opt_t *mem_opt_init(void);
	void mem_fill_scmat(int a, int b, int8_t mat[25]);

	/**
	 * Align a batch of sequences and generate the alignments in the SAM format
	 *
	 * This routine requires $seqs[i].{l_seq,seq,name} and write $seqs[i].sam.
	 * Note that $seqs[i].sam may consist of several SAM lines if the
	 * corresponding sequence has multiple primary hits.
	 *
	 * In the paired-end mode (i.e. MEM_F_PE is set in $opt->flag), query
	 * sequences must be interleaved: $n must be an even number and the 2i-th
	 * sequence and the (2i+1)-th sequence constitute a read pair. In this
	 * mode, there should be enough (typically >50) unique pairs for the
	 * routine to infer the orientation and insert size.
	 *
	 * @param opt    alignment parameters
	 * @param bwt    FM-index of the reference sequence
	 * @param bns    Information of the reference
	 * @param pac    2-bit encoded reference
	 * @param n      number of query sequences
	 * @param seqs   query sequences; $seqs[i].seq/sam to be modified after the call
	 * @param pes0   insert-size info; if NULL, infer from data; if not NULL, it should be an array with 4 elements,
	 *               corresponding to each FF, FR, RF and RR orientation. See mem_pestat() for more info.
	 */
	void mem_process_seqs(const mem_opt_t *opt, const bwt_t *bwt, const bntseq_t *bns, const uint8_t *pac, int64_t n_processed, int n, bseq1_t *seqs, const mem_pestat_t *pes0);

	/**
	 * Find the aligned regions for one query sequence
	 *
	 * Note that this routine does not generate CIGAR. CIGAR should be
	 * generated later by mem_reg2aln() below.
	 *
	 * @param opt    alignment parameters
	 * @param bwt    FM-index of the reference sequence
	 * @param bns    Information of the reference
	 * @param pac    2-bit encoded reference
	 * @param l_seq  length of query sequence
	 * @param seq    query sequence
	 *
	 * @return       list of aligned regions.
	 */
	mem_alnreg_v mem_align1(const mem_opt_t *opt, const bwt_t *bwt, const bntseq_t *bns, const uint8_t *pac, int l_seq, const char *seq);

	/**
	 * Generate CIGAR and forward-strand position from alignment region
	 *
	 * @param opt    alignment parameters
	 * @param bns    Information of the reference
	 * @param pac    2-bit encoded reference
	 * @param l_seq  length of query sequence
	 * @param seq    query sequence
	 * @param ar     one alignment region
	 *
	 * @return       CIGAR, strand, mapping quality and forward-strand position
	 */
	mem_aln_t mem_reg2aln(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_seq, const char *seq, const mem_alnreg_t *ar);
	mem_aln_t mem_reg2aln2(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_seq, const char *seq, const mem_alnreg_t *ar, const char *name);

	/**
	 * Infer the insert size distribution from interleaved alignment regions
	 *
	 * This function can be called after mem_align1(), as long as paired-end
	 * reads are properly interleaved.
	 *
	 * @param opt    alignment parameters
	 * @param l_pac  length of concatenated reference sequence
	 * @param n      number of query sequences; must be an even number
	 * @param regs   region array of size $n; 2i-th and (2i+1)-th elements constitute a pair
	 * @param pes    inferred insert size distribution (output)
	 */
	void mem_pestat(const mem_opt_t *opt, int64_t l_pac, int n, const mem_alnreg_v *regs, mem_pestat_t pes[4]);

#ifdef __cplusplus
}
#endif

#endif
