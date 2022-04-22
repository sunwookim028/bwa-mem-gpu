#include "datadump.hpp"
#include "hashKMer.hpp"

kmers_bucket_t *loadKMerIndex(const char* path){
    kmers_bucket_t *hashTable = loadArray((unsigned long long)pow4(KMER_K), path);
    return hashTable;
}