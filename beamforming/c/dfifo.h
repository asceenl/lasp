// dfifo.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// API of a contiguous fifo buffer of samples.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef DFIFO_H
#define DFIFO_H
#include "types.h"
#include "ascee_math.h"
typedef struct dFifo_s dFifo;

/** 
 * Create a fifo buffer
 *
 * @param nchannels Number of channels to store for
 * @param max_size Maximum size of the queue.
 *
 * @return Pointer to fifo queue (no null ptr)
 */
dFifo* dFifo_create(const us nchannels,
                    const us init_size);


/** 
 * Pushes samples into the fifo.
 *
 * @param fifo dFifo handle
 *
 * @param data data to push. Number of columns should be equal to
 * nchannels.
 */
void dFifo_push(dFifo* fifo,const dmat* data);

/** 
 * Pop samples from the queue
 *
 * @param[in] fifo dFifo handle

 * @param[out] data Pointer to dmat where popped data will be
 * stored. Should have nchannels number of columns. If n_rows is
 * larger than current storage, the queue is emptied.

 * @param[in] keep Keeps a number of samples for the next dFifo_pop(). If
 * keep=0, then no samples will be left. Keep should be smaller than
 * the number of rows in data.
 *
 * @return Number of samples obtained in data.
 */
int dFifo_pop(dFifo* fifo,dmat* data,const us keep);

/** 
 * Returns current size of the fifo
 *
 * @param[in] fifo dFifo handle
 *
 * @return Current size
 */
us dFifo_size(dFifo* fifo);

/** 
 * Free a dFifo object
 *
 * @param[in] fifo dFifo handle.
 */
void dFifo_free(dFifo* fifo);

#endif // DFIFO_H
//////////////////////////////////////////////////////////////////////

