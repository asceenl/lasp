// lasp_worker.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-5)
#include "lasp_worker.h"
#include "lasp_mq.h"
#include "lasp_alloc.h"
#include <pthread.h>
#include "lasp_assert.h"
#include "lasp_tracer.h"

typedef struct Workers_s {
    JobQueue* jq;
    worker_alloc_function w_alloc_fcn;
    worker_function fn;
    worker_free_function w_free_fcn;

    pthread_mutex_t global_data_mutex;
    void* global_data;

    pthread_t worker_threads[LASP_MAX_NUM_THREADS];
    us num_workers;
} Workers;

static void* threadfcn(void* data);

Workers* Workers_create(const us num_workers,
                        JobQueue* jq,
                        worker_alloc_function init_fn,
                        worker_function fn,
                        worker_free_function free_fn,
                        void* thread_global_data) {

    TRACE(15,"Workers_create");
    
    if(num_workers > LASP_MAX_NUM_THREADS) {
        WARN("Number of workers too high in Workers_create");
        return NULL;
    }

    dbgassert(init_fn,NULLPTRDEREF "init_fn");
    dbgassert(fn,NULLPTRDEREF "fn");
    dbgassert(free_fn,NULLPTRDEREF "free_fn");

    Workers* w = a_malloc(sizeof(Workers));
    if(!w){
        WARN(ALLOCFAILED "Workers_create");
        return NULL;
    }

    w->jq = jq;
    w->w_alloc_fcn = init_fn;
    w->fn = fn;
    w->w_free_fcn = free_fn;
    w->global_data = thread_global_data;
    w->num_workers = num_workers;

    /* Initialize thread mutex */
    int rv = pthread_mutex_init(&w->global_data_mutex,NULL);
    if(rv !=0) {
        WARN("Mutex initialization failed");
        return NULL;
    }


    /* Create the threads */
    pthread_t* thread = w->worker_threads;
    for(us i = 0; i < num_workers; i++) {
        TRACE(15,"Creating thread");
        int rv = pthread_create(thread,
                                NULL, /* Thread attributes */
                                threadfcn, /* Function */
                                w);      /* Data */
        if(rv!=0) {
            WARN("Thread creation failed");
            return NULL;
        }
        thread++;
    }

    return w;
    
}
                        
void Workers_free(Workers* w) {
    TRACE(15,"Workers_free");
    dbgassert(w,NULLPTRDEREF "w in Workers_free");
    dbgassert(w->jq,NULLPTRDEREF "w->jq in Workers_free");

    for(us i=0;i<w->num_workers;i++) {
         /* Push the special NULL job. This will make the worker
          * threads stop their execution. */
        JobQueue_push(w->jq,NULL);
    }

    JobQueue_wait_alldone(w->jq);

/* Join the threads */
    pthread_t* thread = w->worker_threads;
    for(us i=0;i<w->num_workers;i++) {
        void* retval;
        if(pthread_join(*thread,&retval)!=0) {
            WARN("Error joining thread!");
        }
        if((retval) != NULL) {
            WARN("Thread returned with error status");
        }
        thread++;
    }

    /* Destroy the global data mutex */
    int rv = pthread_mutex_destroy(&w->global_data_mutex);
    if(rv != 0){
        WARN("Mutex destroy failed. Do not know what to do.");
    }

    /* All threads joined */
    a_free(w);

}

static void* threadfcn(void* thread_global_data) {

    TRACE(15,"Started worker thread function");
    dbgassert(thread_global_data,NULLPTRDEREF "thread_data in"
              " threadfcn");
    Workers* w = (Workers*) thread_global_data;

    JobQueue* jq = w->jq;
    worker_alloc_function walloc = w->w_alloc_fcn;
    worker_free_function wfree = w->w_free_fcn;
    worker_function worker_fn = w->fn;
    void* global_data = w->global_data;

    dbgassert(jq,NULLPTRDEREF "jq in threadfcn");
    dbgassert(walloc,NULLPTRDEREF "walloc in threadfcn");
    dbgassert(wfree,NULLPTRDEREF "wfree in threadfcn");

    int rv = pthread_mutex_lock(&w->global_data_mutex);
    if(rv !=0) {                                
        WARN("Global data mutex lock failed");            
        pthread_exit((void*) 1);
    }

    void* w_data = walloc(global_data);
    if(!w_data) {
        WARN(ALLOCFAILED);
        pthread_exit((void*) 1);
    }
    
    rv = pthread_mutex_unlock(&w->global_data_mutex);
    if(rv !=0) {                                
        WARN("Global data mutex unlock failed");            
        pthread_exit((void*) 1);
    }

    void* job = NULL;
    TRACE(20,"Worker ready");
    while (true) {
        
        TRACE(10,"--------------- START CYCLE -------------");
        job = JobQueue_assign(jq);

        /* Kill the thread for the special NULL job */
        if(!job) break;

        /* Run the worker function */
        rv = worker_fn(w_data,job);
        if(rv!=0) {
            WARN("An error occured during execution of worker function");
            JobQueue_done(jq,job);
            break;
        }

        JobQueue_done(jq,job);
        TRACE(10,"--------------- CYCLE COMPLETE -------------");
    }

    JobQueue_done(jq,job);

    /* Call the cleanup function */
    wfree(w_data);
    TRACE(15,"Exiting thread. Goodbye");
    pthread_exit((void*) NULL);

    /* This return statement is never reached, but added to have a proper return
     * type from this function. */ 
    return NULL;
}


//////////////////////////////////////////////////////////////////////
