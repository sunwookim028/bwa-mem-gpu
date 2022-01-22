#ifndef BATCH_CONFIG_H
#define BATCH_CONFIG_H


#define SEQ_MAXLEN 500				// max length of a seq
#define SEQ_MAX_COUNT 40000			// max number of seqs 
#define SEQ_NAME_LIMIT 2000000 		// chunk size of name
#define SEQ_COMMENT_LIMIT 40000000	// chunk size of comment
#define SEQ_LIMIT SEQ_MAXLEN*SEQ_MAX_COUNT	// chunk size of seq
#define SEQ_QUAL_LIMIT 40000000		// chunk size of qual
#define SEQ_SAM_LIMIT 200000000		// chunk size of sam output


#endif