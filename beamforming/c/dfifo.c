// dfifo.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// Implementation of the dFifo queue
//////////////////////////////////////////////////////////////////////

#include "dfifo.h"
#define DFIFO_QUEUE_MAX_BLOCKS (50)

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
dFifo* dFifo_create(const us nchannels,
                    const us max_size) {

    fsTRACE(15);
    dFifo* fifo = a_malloc(sizeof(dFifo));
    fifo->queue = dmat_alloc(max_size,nchannels);
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
int dFifo_push(dFifo* fifo,const dmat* data) {
    fsTRACE(15);
    dbgassert(fifo && data, NULLPTRDEREF);
    dbgassert(data->n_cols == fifo->queue.n_cols,
              "Invalid number of columns in data");


    dmat queue = fifo->queue;
    const us max_size = queue.n_rows;
    const us nchannels = queue.n_cols;

    us* start_row = &fifo->start_row;
    us* end_row = &fifo->end_row;
    const us added_size = data->n_rows;
    const us size_before = dFifo_size(fifo);

    if(added_size + dFifo_size(fifo) > max_size) {
        return FIFO_QUEUE_FULL;
    }

    if(*end_row + added_size > max_size) {
        if(size_before != 0) {
            /* Shift the samples to the front of the queue. TODO: this
             * might not be the most optimal implementation (but it is the
             * most simple). */
            TRACE(15,"Shift samples back in buffer");
            uVARTRACE(15,size_before);
            dmat tmp = dmat_alloc(size_before,nchannels);
            TRACE(15,"SFSG");
            dmat_copy_rows(&tmp,
                           &queue,
                           0,
                           *start_row,
                           size_before);
            TRACE(15,"SFSG");
            dmat_copy_rows(&queue,
                           &tmp,
                           0,0,size_before);

            *end_row -= *start_row;
            *start_row = 0;

            dmat_free(&tmp);
        }
        else {
            *start_row = 0;
            *end_row = 0;
        }
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
    return SUCCESS;
}
int dFifo_pop(dFifo* fifo,dmat* data,const us keep) {
    fsTRACE(15);
    dbgassert(fifo && data,NULLPTRDEREF);
    dbgassert(data->n_cols == fifo->queue.n_cols,
              "Invalid number of columns in data");
    dbgassert(keep < data->n_rows, "Number of samples to keep should"
              " be smaller than requested number of samples");

    us* start_row = &fifo->start_row;
    us* end_row = &fifo->end_row;
    us cur_contents = dFifo_size(fifo);
    us requested = data->n_rows;
    
    us obtained = requested > cur_contents ? cur_contents : requested;
    dbgassert(obtained > keep,"Number of samples to keep should be"
              " smaller than requested number of samples");

    uVARTRACE(15,requested);
    uVARTRACE(15,obtained);
    uVARTRACE(15,*start_row);
    uVARTRACE(15,*end_row);
    
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
