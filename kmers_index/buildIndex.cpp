#include "hashKMer.hpp"
#include <iostream>
#include <string.h>
#include <random>
#include <cstdlib>
#include "../bwa.h"
#include "loadKMerIndex.cpp"
#include <stdio.h>
#include <locale.h>





int main(int argc, char const *argv[])
{
    setlocale(LC_ALL, ""); /* use user selected locale */
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 3); // define the range

    // load bwt index from disk
    bwaidx_t *idx;
    if ((idx = bwa_idx_load(argv[optind], BWA_IDX_BWT|BWA_IDX_BNS)) == 0) {
        std::cerr << "can't load bwt index!" << std::endl;
        return 1;
    }
    
    // create hash table
    kmers_bucket_t *hashTable = createHashKTable(idx->bwt);

    // dump hash table to binary file
    dumpArray(hashTable, pow4(KMER_K), "hashTable");

    // print hash table to std out
    kmers_bucket_t *hashTable2 = loadKMerIndex("./hashTable");

    for (int i=0; i<pow4(KMER_K); i++)
        printf("%9s %'14lu %'14lu %'14lu\n", inverseHashK(i), hashTable2[i].x[0], hashTable2[i].x[1], hashTable2[i].x[2]);

    /* code */
    return 0;
}
