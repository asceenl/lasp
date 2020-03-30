// mq.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// Multithreaded job queue implementation
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef MQ_H
#define MQ_H

typedef struct JobQueue_s JobQueue;

/** 
 * Allocate a new job queue.
 *
 * @param max_msg Maximum number of jobs that can be put in the
 * queue.
 *
 * @return Pointer to new JobQueue instance. NULL on error.
 */
JobQueue* JobQueue_alloc(const us max_msg);

/** 
 * Free an existing job queue. If it is not empty and threads are
 * still waiting for jobs, the behaviour is undefined. So please
 * make sure all threads are done before free'ing the queue.
 *
 * @param jq: JobQueue to free
 */
void JobQueue_free(JobQueue* jq);

/** 
 * Pops a job from the queue. Waits indefinitely until some job is
 * available.
 *
 * @param jq: JobQueue handle
 * @return Pointer to the job, NULL on error.
 */
void* JobQueue_assign(JobQueue* jq);

/** 
 * Tell the queue the job that has been popped is done. Only after
 * this function call, the job is really removed from the queue.
 *
 * @param jq: JobQueue handle
 * @param job 
 */
void JobQueue_done(JobQueue* jq,void* job);

/** 
 * A push on the job queue will notify one a single thread that is
 * blocked waiting in the JobQueue_assign() function. If the job
 * queue is full, however all waiters will be signaled and the
 * function will block until there is some space in the job queue.
 *
 * @param jp JobQueue
 * @param job_ptr Pointer to job to be done
 * @return 0 on success.
 */
int JobQueue_push(JobQueue* jp,void* job_ptr);

/** 
 * Wait until the job queue is empty. Please use this function with
 * caution, as it will block indefinitely in case the queue never gets
 * empty. The purpose of this function is to let the main thread wait
 * until all task workers are finished.
 *
 */
void JobQueue_wait_alldone(JobQueue*);

#endif // MQ_H
//////////////////////////////////////////////////////////////////////
