#include "hashKMer.hpp"
#include <iostream>
#include <string.h>
#include <random>
#include <cstdlib>
#include "../bwa.h"
#include "loadIndex.hpp"



int main(int argc, char const *argv[])
{
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, 3); // define the range

    // load index from disk
    bwaidx_t *idx;
    if ((idx = bwa_idx_load(argv[optind], BWA_IDX_BWT|BWA_IDX_BNS)) == 0) {
        std::cerr << "can't load index!" << std::endl;
        return 1;
    }
    
    // create hash table
    bucket_t *hashTable = createHashKTable(idx->bwt);

    // dump hash table to binary file
    dumpArray(hashTable, pow4(KMER_K), "hashTable");

    // print hash table to std out
    bucket_t *hashTable2 = loadIndex();

    for (int i=0; i<pow4(KMER_K); i++)
        std::cout << inverseHashK(i) << "  " << hashTable2[i].x[2] << std::endl;

    /* code */
    return 0;
}
