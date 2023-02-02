#include "superbatch_process.h"
#include "minibatch_process.h"
#include "streams.cuh"
#include <future>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <climits>
using namespace std;
ofstream perf_profile_file("perf_profile.txt");

/**
 * @brief initiate memory for a super batch
 * @return superbatch_data_t*
 */
static superbatch_data_t *newSuperBatchData()
{
    superbatch_data_t *batch = (superbatch_data_t *)malloc(sizeof(superbatch_data_t));
    batch->n_reads = 0;
    // init memory for reads in the batch
    batch->reads = (bseq1_t *)malloc(SB_MAX_COUNT * sizeof(bseq1_t));
    batch->name = (char *)malloc(SB_NAME_LIMIT);
    batch->seqs = (char *)malloc(SB_SEQ_LIMIT);
    batch->comment = (char *)malloc(SB_COMMENT_LIMIT);
    batch->qual = (char *)malloc(SB_QUAL_LIMIT);
    if (batch->reads == nullptr || batch->name == nullptr || batch->seqs == nullptr || batch->comment == nullptr || batch->qual == nullptr)
    {
        fprintf(stderr, "[M::%-25s] can't malloc superbatch\n", __func__);
        exit(1);
    }
    if (bwa_verbose >= 3)
    {
        double nGB_allocated = (double)(SB_MAX_COUNT * sizeof(bseq1_t) + SB_NAME_LIMIT + SB_SEQ_LIMIT + SB_COMMENT_LIMIT + SB_QUAL_LIMIT) / (1024ULL * 1024ULL * 1024ULL);
        fprintf(stderr, "[M::%-25s] allocated %.2f GB on host for superbatch\n", __func__, nGB_allocated);
    }
    return batch;
}

/**
 * @brief remove data from a superbatch data set
 */
static void resetSuperBatchData(superbatch_data_t *data)
{
    data->n_reads = 0;
    data->name_size = 0;
    data->comment_size = 0;
    data->seqs_size = 0;
    data->qual_size = 0;
}

/**
 * @brief compare 2 reads a and b.
 * @return int positive if a > b, negative if a < b, 0 if a == b
 */
static int compareReads(const void *a, const void *b)
{
    char *a_key = ((bseq1_t *)a)->seq;
    char *b_key = ((bseq1_t *)b)->seq;
    return strncmp(a_key, b_key, 500);
}

/**
 * @brief sort reads lexicographically
 */
static void sortReads(bseq1_t *reads, int n_reads)
{
    qsort(reads, n_reads, sizeof(bseq1_t), compareReads);
}

/**
 * @brief
 *
 * @param ks
 * @param ks2
 * @param actual_chunk_size
 * @param copy_comment
 * @param transfer_data
 * @return int number of reads loaded from file
 */
static int loadInputSuperBatch(kseq_t *ks, kseq_t *ks2, int actual_chunk_size, int copy_comment, superbatch_data_t *transfer_data)
{
    struct timespec timing_start, timing_stop; // variables for printing timings
    if (bwa_verbose >= 3)
        clock_gettime(CLOCK_MONOTONIC_RAW, &timing_start);
    int64_t size = 0;
    int n_seqs_read;
    bseq_read2(actual_chunk_size, &n_seqs_read, ks, ks2, transfer_data); // this will write to transfer_data
    bseq1_t *reads = transfer_data->reads;
    transfer_data->n_reads = n_seqs_read;
    if (n_seqs_read == 0)
    {
        return 0;
    }
    if (copy_comment)
        for (int i = 0; i < n_seqs_read; ++i)
        {
            reads[i].comment = 0;
        }

    if (bwa_verbose >= 3)
        clock_gettime(CLOCK_MONOTONIC_RAW, &timing_stop);
    if (bwa_verbose >= 3)
        fprintf(stderr, "[M::%-25s] ***load %'d reads from disk took %lu ms\n", __func__, n_seqs_read, (timing_stop.tv_nsec - timing_start.tv_nsec) / 1000000);

    // sort reads
    if (bwa_verbose >= 3)
        clock_gettime(CLOCK_MONOTONIC_RAW, &timing_start);
    // sortReads(reads, n_seqs_read);
    if (bwa_verbose >= 3)
        clock_gettime(CLOCK_MONOTONIC_RAW, &timing_stop);
    if (bwa_verbose >= 3)
        fprintf(stderr, "[M::%-25s] ***sort reads took %lu ms\n", __func__, (timing_stop.tv_nsec - timing_start.tv_nsec) / 1000000);

    return n_seqs_read;
}

static void processSuperBatch(superbatch_data_t *data, transfer_data_t *mini_transfer, process_data_t *mini_process)
{
    miniBatchMain(data, mini_transfer, mini_process);
}

/**
 * @brief process all data in fasta files using super batches
 *
 * @param aux top-level data on this program: input fasta files, indexes, mapping parameters.
 */
void superBatchMain(ktp_aux_t *aux)
{
    perf_profile_file << "batch,SMEM_CHN(ms),BSW(ms),SAM(ms)\n";

    // init memory for 2 superbatches, 1 for processing and 1 for transfer
    superbatch_data_t *super_process = newSuperBatchData();
    superbatch_data_t *super_transfer = newSuperBatchData();
    // init memory for minibatches so we don't have to do this repeatedly, 1 for processing and 1 for transfer
    process_data_t *mini_process = newProcess(aux->opt, aux->pes0, aux->idx->bwt, aux->idx->bns, aux->idx->pac, aux->kmerHashTab);
    transfer_data_t *mini_transfer = newTransfer();

    bool first_batch = true;
    // process until there is no more reads to process
    // except first batch where there is not yet data
    while (first_batch || super_process->n_reads != 0)
    {
        // async process current batch
        auto processAsync = std::async(std::launch::async, processSuperBatch, super_process, mini_transfer, mini_process); // does nothing if process_data->n_seqs is 0
        // async input next batch (process 0)
        auto inputAsync = std::async(std::launch::async, loadInputSuperBatch, aux->ks, aux->ks2, INT_MAX, aux->copy_comment, super_transfer);

        inputAsync.wait();
        processAsync.wait();

        aux->n_processed += super_process->n_reads;
        // verbose
        if (aux->n_processed > 0)
            fprintf(stderr, "[M::%-25s] **** superbatch %'ld finished \n", __func__, aux->n_processed);

        // swap the two data sets for next iteration processing
        auto tmp = super_process;
        super_process = super_transfer;
        super_transfer = tmp;
        resetSuperBatchData(super_transfer);
        first_batch = false;
    }
}
