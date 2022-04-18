#ifndef HASHKMER_HPP
#define HASHKMER_HPP

#include <cstdlib>
#include "../bwt.h"
#include <iostream>

#define KMER_K 9

// convert A,C,G,T,N to 0,1,2,3,4
const int __nst_nt4_table__[256] = {
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

#define charToInt(c) (__nst_nt4_table__[(int)c])
#define intToChar(x) ("ACGTN"[(x)])
#define pow4(x) (1<<(2*(x)))  // 4^x


int hashK(const char* s){
    int out = 0;
    for (int i=0; i<KMER_K; i++){
        if (s[i]=='N' || s[i]=='n') return -1;
        out += charToInt(s[i])*pow4(KMER_K-1-i);
    }
    return out;
}

char* inverseHashK(int x){
	char* out = (char*)malloc((KMER_K+1)*sizeof(char));
	for (int i=0; i<KMER_K; i++){
		if (x==0) out[KMER_K-1-i] = intToChar(0);
		else out[KMER_K-1-i] = intToChar(x%4);
		x = x/4;
	}
	out[KMER_K] = '\0';
	return out;
}


typedef struct {
	bwtint_t x[3]; // same as first 3 elements on bwtintv_t
	// bwtintv_t.info not included here because it contains length of match, which is always KMER_K in this case
} bucket_t;


// hash table as array
// (arrayIndex = hashValue) --> bwt interval bucket_t
bucket_t* createHashKTable(const bwt_t *bwt){
	bucket_t* out = (bucket_t*)malloc(pow4(KMER_K)*sizeof(bucket_t));
	for (int hashValue=0; hashValue<pow4(KMER_K); hashValue++){
		char *read = inverseHashK(hashValue);	// K-length string for finding intervals
		bwtintv_t ik, ok[4];
		bwt_set_intv(bwt, charToInt(read[0]), ik); // set ik = the initial interval of the first base
		for (int i=1; i<KMER_K; i++){	// forward extend
			if (ik.x[2] < 1) break; // no match
			char cb = 3 - charToInt(read[i]);	// complement of next base
			bwt_extend(bwt, &ik, ok, 0);
			ik = ok[cb];
		}
		// save result
		out[hashValue].x[0] = ik.x[0];
		out[hashValue].x[1] = ik.x[1];
		out[hashValue].x[2] = ik.x[2];
	}
	
	return out;
}

#endif