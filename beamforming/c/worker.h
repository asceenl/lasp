// worker.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Provides a clean interface to a pool of worker
// threads. This class is used to easily interface with worker threads
// by just providing a simple function to be called by a worker thread
// on the push of a new job.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef WORKER_H
#define WORKER_H
#include "types.h"

typedef struct Workers_s Workers;
typedef struct JobQueue_s JobQueue;

typedef void* (*worker_alloc_function)(void* global_data);
typedef int (*worker_function)(void* worker_data,void* job);
typedef void (*worker_free_function)(void* worker_data);

/** 
 * Create a pool of worker threads, that pull jobs from the job queue
 * and perform the action. 
 *
 * @param num_workers Number of worker threads to create
 *
 * @param jq JobQueue. JobQueue where jobs for the workers are
 * pushed. Should stay valid as long as the Workers are alive.
 *
 * @param worker_alloc_function Function pointer to the function that
 * will be called right after the thread has been created. The worker
 * alloc function will get a pointer to the thread global data. This
 * data will be given to each thread during initialization. Using a
 * mutex to avoid race conditions on this global data.

 * @param fn Worker function that performs the action on the
 * data. Will be called every time a job is available from the
 * JobQueue. Should have a return code of 0 on success.
 *
 * @param worker_free_function Cleanup function that is called on
 * exit. 
 *
 * @return Pointer to Workers handle. NULL on error.
 */
Workers* Workers_create(const us num_workers,
                        JobQueue* jq,
                        worker_alloc_function init_fn,
                        worker_function fn,
                        worker_free_function free_fn,
                        void* thread_global_data);

/** 
 * Free the pool of workers.
 *
 * @param w 
 */
void Workers_free(Workers* w);

#endif // WORKER_H
//////////////////////////////////////////////////////////////////////
