#ifndef MINIBATCH_PROCESS_H
#define MINIBATCH_PROCESS_H

#include "superbatch_process.h"

#ifdef __cplusplus
extern "C"
{
#endif

	void miniBatchMain(superbatch_data_t *data, transfer_data_t *transfer_data, process_data_t *process_data);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif