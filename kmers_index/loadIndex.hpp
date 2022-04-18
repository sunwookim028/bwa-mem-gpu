#include "datadump.hpp"
#include "hashKMer.hpp"

bucket_t *loadIndex(){
    bucket_t *hashTable = loadArray((unsigned long long)pow4(8), "hashTable");
    return hashTable;
}