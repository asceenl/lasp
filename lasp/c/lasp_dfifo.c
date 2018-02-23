// lasp_dfifo.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// Implementation of the dFifo queue
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-5)
#include "lasp_dfifo.h"

typedef struct dFifo_s {
    dmat queue;
    us start_row;
    us end_row;
} dFifo;
us dFifo_size(dFifo* fifo) {
    fsTRACE(15);
    dbgassert(fifo,NULLPTRDEREF);
    dbgassert(fifo->start_row <= fifo->end_row,"BUG");
    feTRACE(15);
    return fifo->end_row-fifo->start_row;
}

/** 
 * Change the max size of the dFifo to a new max size specified. Max size
 * should be larger than fifo size. Resets start row to 0
 *
 * @param fifo 
 * @param new_size 
 */
static void dFifo_change_maxsize(dFifo* fifo,const us new_max_size) {
    fsTRACE(30);
    dmat old_queue = fifo->queue;
    
    dbgassert(new_max_size >= dFifo_size(fifo),"BUG");
    const us size = dFifo_size(fifo);

    dmat new_queue = dmat_alloc(new_max_size,old_queue.n_cols);
    if(size > 0) {
        dmat_copy_rows(&new_queue,
                       &old_queue,
                       0,
                       fifo->start_row,
                       size);
    }
 
    dmat_free(&old_queue);
    fifo->queue = new_queue;
    fifo->end_row -= fifo->start_row;
    fifo->start_row = 0;

   feTRACE(30);                   
}

dFifo* dFifo_create(const us nchannels,
                    const us init_size) {

    fsTRACE(15);
    dFifo* fifo = a_malloc(sizeof(dFifo));
    fifo->queue = dmat_alloc(init_size,nchannels);
    fifo->start_row = 0;
    fifo->end_row = 0;
    feTRACE(15);
    return fifo;
}
void dFifo_free(dFifo* fifo) {
    fsTRACE(15);
    dmat_free(&fifo->queue);
    a_free(fifo);
    feTRACE(15);
}
void dFifo_push(dFifo* fifo,const dmat* data) {
    fsTRACE(15);
    dbgassert(fifo && data, NULLPTRDEREF);
    dbgassert(data->n_cols == fifo->queue.n_cols,
              "Invalid number of columns in data");


    const us added_size = data->n_rows;

    dmat queue = fifo->queue;
    const us max_size = queue.n_rows;

    us* end_row = &fifo->end_row;

    if(added_size + dFifo_size(fifo) > max_size) {
        dFifo_change_maxsize(fifo,2*(max_size+added_size));

        /* Now the stack of this function is not valid anymore. Best
         * thing to do is restart the function. */
         dFifo_push(fifo,data);
         feTRACE(15);
         return;
    }
    else if(*end_row + added_size > max_size) {
        dFifo_change_maxsize(fifo,max_size);
        /* Now the stack of this function is not valid anymore. Best
         * thing to do is restart the function. */
        dFifo_push(fifo,data);
        feTRACE(15);
        return;

    }

    /* Now, copy samples */
    dmat_copy_rows(&queue,      /* to */
                   data,        /* from */
                   *end_row, /* startrow_to */
                   0,                /* startrow_from */
                   added_size);      /* n_rows */

    /* Increase the size */
    *end_row += added_size;

    feTRACE(15);
}
int dFifo_pop(dFifo* fifo,dmat* data,const us keep) {
    fsTRACE(15);
    dbgassert(fifo && data,NULLPTRDEREF);
    dbgassert(data->n_cols == fifo->queue.n_cols,
              "Invalid number of columns in data");
    dbgassert(keep < data->n_rows, "Number of samples to keep should"
              " be smaller than requested number of samples");

    us* start_row = &fifo->start_row;
    us cur_size = dFifo_size(fifo);
    us requested = data->n_rows;
    
    us obtained = requested > cur_size ? cur_size : requested;
    dbgassert(obtained > keep,"Number of samples to keep should be"
              " smaller than requested number of samples");

    
    dmat_copy_rows(data,
                   &fifo->queue,
                   0,
                   *start_row,
                   obtained);
    
    *start_row += obtained - keep;

    feTRACE(15);
    return (int) obtained;
}
//////////////////////////////////////////////////////////////////////
