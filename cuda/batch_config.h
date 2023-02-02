#ifndef BATCH_CONFIG_H
#define BATCH_CONFIG_H


#define SEQ_MAXLEN 301// max length of a seq we want to process

// mini-batch config
#define MB_MAX_COUNT 20000                     // max number of reads
#define MB_NAME_LIMIT MB_MAX_COUNT * 100       // chunk size of name
#define MB_COMMENT_LIMIT MB_MAX_COUNT * 100    // chunk size of comment
#define MB_SEQ_LIMIT MB_MAX_COUNT *SEQ_MAXLEN  // chunk size of seq
#define MB_QUAL_LIMIT MB_MAX_COUNT *SEQ_MAXLEN // chunk size of qual
#define MB_SAM_LIMIT MB_MAX_COUNT * 5000       // chunk size of sam output

// super-batch config
#define SB_MAX_COUNT 1000000                   // max number of reads
#define SB_NAME_LIMIT (unsigned long)SB_MAX_COUNT * 100       // chunk size of name
#define SB_COMMENT_LIMIT (unsigned long)SB_MAX_COUNT * 100    // chunk size of comment
#define SB_SEQ_LIMIT (unsigned long)SB_MAX_COUNT *SEQ_MAXLEN  // chunk size of seq
#define SB_QUAL_LIMIT (unsigned long)SB_MAX_COUNT *SEQ_MAXLEN // chunk size of qual


#endif