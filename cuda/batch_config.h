#define SEQ_MAXLEN 500				// max length of a seq
#define SEQ_MAX_COUNT 40000			// max number of seqs 
#define SEQ_NAME_LIMIT 2000000 		// chunk size of name
#define SEQ_COMMENT_LIMIT 40000000	// chunk size of comment
#define SEQ_LIMIT SEQ_MAXLEN*SEQ_MAX_COUNT	// chunk size of seq
#define SEQ_QUAL_LIMIT 40000000		// chunk size of qual
#define SEQ_SAM_LIMIT 200000000		// chunk size of sam output

/* global pointers to keep track of amount of data on loaded seqs */
extern bseq1_t *preallocated_seqs;		// a chunk of SEQ_MAX_COUNT seqs on host's pinned memory
extern bseq1_t *d_preallocated_seqs;	// same chunk on device
extern char *seq_name_ptr, *seq_comment_ptr, *seq_ptr, *seq_qual_ptr, *seq_sam_ptr;		// pointers to chunks on host
extern char *d_seq_name_ptr, *d_seq_comment_ptr, *d_seq_ptr, *d_seq_qual_ptr;			// pointers to chunks on device
extern int seq_name_offset, seq_comment_offset, seq_offset, seq_qual_offset;			// offsets on the chunks
