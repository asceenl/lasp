// mq.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description: Message queue implementation using a linked
// list. Using mutexes and condition variables to sync access between
// different threads.
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-6)
#include "types.h"
#include "tracer.h"
#include "ascee_assert.h"
#include "ascee_alloc.h"
#include "mq.h"
#include <pthread.h>

#ifdef __linux__
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#endif

typedef struct {
    void* job_ptr;
    bool running;
    bool ready;
} Job;

typedef struct JobQueue_s {
    pthread_mutex_t mutex;    
    pthread_cond_t cv_plus;          /**< Condition variable for the
                                      * "workers". */ 
    pthread_cond_t cv_minus;          /**< Condition variable for the
                                       * main thread. */   

    Job* jobs; /**< Pointer to job vector */
    us max_jobs;               /**< Stores the maximum number of
                                *   items */
} JobQueue;

static us count_jobs(JobQueue* jq) {
    fsTRACE(15);
    us njobs = 0;
    for(us i=0;i<jq->max_jobs;i++){
        if(jq->jobs[i].ready)
            njobs++;
    }
    return njobs;
}
static Job* get_ready_job(JobQueue* jq) {
    fsTRACE(15);
    Job* j = jq->jobs;
    for(us i=0;i<jq->max_jobs;i++){
        if(j->ready && !j->running)
            return j;
        j++;
    }
    return NULL;
}
void print_job_queue(JobQueue* jq) {
    fsTRACE(15);
    for(us i=0;i<jq->max_jobs;i++) {
        printf("Job %zu", i);
        if(jq->jobs[i].ready)
            printf(" available");
        if(jq->jobs[i].running)
            printf(" running");

        printf(" - ptr %zu\n", (us) jq->jobs[i].job_ptr);

    }
    feTRACE(15);
}


#define LOCK_MUTEX                                              \
    /* Lock the mutex to let the threads wait initially */      \
    int rv = pthread_mutex_lock(&jq->mutex);                    \
    if(rv !=0) {                                                \
        WARN("Mutex lock failed");                              \
    }

#define UNLOCK_MUTEX                            \
    rv = pthread_mutex_unlock(&jq->mutex);      \
    if(rv !=0) {                                \
        WARN("Mutex unlock failed");            \
    }

JobQueue* JobQueue_alloc(const us max_jobs) {
    TRACE(15,"JobQueue_alloc");
    if(max_jobs > ASCEE_MAX_NUM_CHANNELS) {
        WARN("Max jobs restricted to ASCEE_MAX_NUM_CHANNELS");
        return NULL;
    }
    JobQueue* jq = a_malloc(sizeof(JobQueue));
    

    if(!jq) {
        WARN("Allocation of JobQueue failed");
        return NULL;
    }
    jq->max_jobs = max_jobs;

    jq->jobs = a_malloc(max_jobs*sizeof(Job));
    if(!jq->jobs) {
        WARN("Allocation of JobQueue jobs failed");
        return NULL;
    }

    Job* j = jq->jobs;
    for(us jindex=0;jindex<max_jobs;jindex++) {
        j->job_ptr = NULL;
        j->ready = false;
        j->running = false;
        j++;
    }

    /* Initialize thread mutex */
    int rv = pthread_mutex_init(&jq->mutex,NULL);
    if(rv !=0) {
        WARN("Mutex initialization failed");
        return NULL;
    }
    rv = pthread_cond_init(&jq->cv_plus,NULL);
    if(rv !=0) {
        WARN("Condition variable initialization failed");
        return NULL;
    }

    rv = pthread_cond_init(&jq->cv_minus,NULL);
    if(rv !=0) {
        WARN("Condition variable initialization failed");
        return NULL;
    }

    /* print_job_queue(jq); */
    return jq;
}

void JobQueue_free(JobQueue* jq) {

    TRACE(15,"JobQueue_free");
    dbgassert(jq,NULLPTRDEREF "jq in JobQueue_free");
    
    int rv;
    
    if(count_jobs(jq) != 0) {
        WARN("Job queue not empty!");
    }

    a_free(jq->jobs);

    /* Destroy the mutexes and condition variables */
    rv = pthread_mutex_destroy(&jq->mutex);
    if(rv != 0){
        WARN("Mutex destroy failed. Do not know what to do.");
    }

    rv = pthread_cond_destroy(&jq->cv_plus);
    if(rv != 0){
        WARN("Condition variable destruction failed. "
             "Do not know what to do.");
    }

    rv = pthread_cond_destroy(&jq->cv_minus);
    if(rv != 0){
        WARN("Condition variable destruction failed. "
             "Do not know what to do.");
    }

}

int JobQueue_push(JobQueue* jq,void* job_ptr) {
    
    TRACE(15,"JobQueue_push");
    dbgassert(jq,NULLPTRDEREF "jq in JobQueue_push");

    /* print_job_queue(jq); */
    /* uVARTRACE(15,(us) job_ptr); */
    
    LOCK_MUTEX;
    
    us max_jobs = jq->max_jobs;

    /* Check if queue is full */
    while(count_jobs(jq) == max_jobs) {

        WARN("Queue full. Wait until some jobs are done.");
        rv = pthread_cond_wait(&jq->cv_minus,&jq->mutex);
        if(rv !=0) {
            WARN("Condition variable wait failed");
        }
    }

    dbgassert(count_jobs(jq) != max_jobs,
              "Queue cannot be full!");

    /* Queue is not full try to find a place, fill it */
    Job* j = jq->jobs;
    us i;
    for(i=0;i<max_jobs;i++) {
        if(j->ready == false ) {
            dbgassert(j->job_ptr==NULL,"Job ptr should be 0");
            dbgassert(j->ready==false,"Job cannot be assigned");            
            break;
        }
        j++;
    }
    dbgassert(i!=jq->max_jobs,"Should have found a job!");

    j->job_ptr = job_ptr;
    j->ready = true;

    /* Notify worker threads that a new job has arrived */
    if(count_jobs(jq) == max_jobs) {
        /* Notify ALL threads. Action required! */
        rv = pthread_cond_broadcast(&jq->cv_plus);
        if(rv !=0) {
            WARN("Condition variable broadcast failed");
        }

    } else {
        /* Notify some thread that there has been some change to
         * the Queue */
        rv = pthread_cond_signal(&jq->cv_plus);
        if(rv !=0) {
            WARN("Condition variable signal failed");
        }

    }

    /* print_job_queue(jq); */

    UNLOCK_MUTEX;

    return SUCCESS;
}
void* JobQueue_assign(JobQueue* jq) {

    TRACE(15,"JobQueue_assign");

    LOCK_MUTEX;

    /* Wait until a job is available */
    Job* j;
    while ((j=get_ready_job(jq))==NULL) {

        TRACE(15,"JobQueue_assign: no ready job");
        pthread_cond_wait(&jq->cv_plus,&jq->mutex);

    }

    TRACE(16,"JobQueue_assign: found ready job. Assigned to:");
    #ifdef ASCEE_DEBUG
    #ifdef __linux__
    
    pid_t tid = syscall(SYS_gettid);    
    iVARTRACE(16,tid);    
    #endif // __linux__
    
    
    #endif //  ASCEE_DEBUG


    /* print_job_queue(jq); */
    /* Find a job from the queue, assign it and return it */
    j->running = true;
    
    if(count_jobs(jq) > 1) {
        /* Signal different thread that there is more work to do */
        rv = pthread_cond_signal(&jq->cv_plus);
        if(rv !=0) {
            WARN("Condition variable broadcast failed");
        }
    }

    UNLOCK_MUTEX;

    TRACE(15,"End JobQueue_assign");

    return j->job_ptr;
}
void JobQueue_done(JobQueue* jq,void* job_ptr) {

    TRACE(15,"JobQueue_done");
    dbgassert(jq,NULLPTRDEREF "jq in JobQueue_done");

    LOCK_MUTEX;
    
    /* print_job_queue(jq); */

    /* Find the job from the queue, belonging to the job_ptr */
    Job* j=jq->jobs;
    us i;
    for(i=0;i<jq->max_jobs;i++) {
        iVARTRACE(10,i);
        if(j->ready && j->running && j->job_ptr == job_ptr) {
            TRACE(15,"Found the job that has been done:");
            j->ready = false;
            j->job_ptr = NULL;
            j->running = false;
            break;
        }
        j++;
    }

    /* print_job_queue(jq); */
    
    /* Job done, broadcast this */
    rv = pthread_cond_signal(&jq->cv_minus);
    if(rv !=0) {
        WARN("Condition variable broadcast failed");
    }

    UNLOCK_MUTEX;
}

void JobQueue_wait_alldone(JobQueue* jq) {
    TRACE(15,"JobQueue_wait_alldone");
    dbgassert(jq,NULLPTRDEREF "jq in JobQueue_wait_alldone");

    LOCK_MUTEX;

    /* Wait until number of jobs is 0 */
    while (count_jobs(jq)!=0) {
        
        if(rv !=0) {
            WARN("Condition variable broadcast failed");
        }
        
        pthread_cond_wait(&jq->cv_minus,&jq->mutex);
    }
    
    UNLOCK_MUTEX;

}


//////////////////////////////////////////////////////////////////////
