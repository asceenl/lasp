// test_bf.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#include "lasp_worker.h"
#include "lasp_mq.h"
#include "lasp_tracer.h"
#include "lasp_assert.h"
#include <unistd.h>
static void* walloc(void*);
static int worker(void*,void*);
static void wfree(void*);


int main() {

    fsTRACE(15);

    iVARTRACE(15,getTracerLevel());

    us njobs = 4;
    JobQueue* jq = JobQueue_alloc(njobs);
    dbgassert(jq,NULLPTRDEREF);

    Workers* w = Workers_create(njobs,
                                jq,
                                walloc,
                                worker,
                                wfree,
                                (void*) 101);
    dbgassert(jq,NULLPTRDEREF);

    for(us i=0; i< njobs; i++) {
        iVARTRACE(15,i);
        JobQueue_push(jq,(void*) i+1);
    }

    JobQueue_wait_alldone(jq);
    Workers_free(w);
    JobQueue_free(jq);

    return 0;
}
static void* walloc(void* data) {
    TRACE(15,"WALLOC");
    uVARTRACE(15,(us) data);
    return (void*) 1;
}

static int worker(void* w_data,void* tj) {
    
    TRACE(15,"worker");
    
    sleep(4);

    return 0;

}
static void wfree(void* w_data) {
    TRACE(15,"wfree");
}



//////////////////////////////////////////////////////////////////////


