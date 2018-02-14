// aps.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////

#include "aps.h"
#include "ps.h"
#include "ascee_alg.h"
#include "dfifo.h"

/* Multiplication factor for the maximum size of the fifo queue. This
 * factor is multiplied by nfft to obtain the maximum size of the
 * fifo. */
#define FIFO_SIZE_MULT (5)

typedef struct AvPowerSpectra_s {
    us nfft, nchannels;
    us overlap;                 /* Number of samples to overlap */

    us naverages;               /* Counter that counts the number of
                                 * averages taken for the computation
                                 * of this averaged power spectra. */
    
    dmat buffer;                /**< Buffer storage of some of the
                                 * previous samples. Number of rows is
                                 * equal to nfft. */


    dFifo* fifo;          /* Sample fifo storage */

    cmat ps_storage;           /**< Here we store the averaged
                                * results for each Cross-power
                                * spectra computed so far. */
    
    cmat ps_single;             /**< This is the work area for a
                                 * PowerSpectra computation on a
                                 * single block */

    PowerSpectra* ps;           /**< Pointer to underlying
                                 * PowerSpectra calculator. */
    
} AvPowerSpectra;

void AvPowerSpectra_free(AvPowerSpectra* aps) {
    fsTRACE(15);

    PowerSpectra_free(aps->ps);
    dFifo_free(aps->fifo);
    dmat_free(&aps->buffer);
    cmat_free(&aps->ps_storage);
    cmat_free(&aps->ps_single);
    a_free(aps);
    
    feTRACE(15);
}

AvPowerSpectra* AvPowerSpectra_alloc(const us nfft,
                                     const us nchannels,
                                     const d overlap_percentage,
                                     const WindowType wt) {
    
    fsTRACE(15);
    
    /* Check nfft */
    if(nfft==0 || nfft % 2 != 0 || nfft > ASCEE_MAX_NFFT) {
        WARN("Invalid nfft");
        feTRACE(15);
        return NULL;
    }

    /* Check overlap percentage */
    if(overlap_percentage < 0. || overlap_percentage >= 100.) {
        WARN("Invalid overlap percentage");
        feTRACE(15);
        return NULL;
    }
    
    /* Compute and check overlap offset */
    us overlap = (us) (overlap_percentage*((d) nfft)/100);
    if(overlap == nfft) {
        WARN("Overlap percentage results in full overlap, decreasing overlap.");
        overlap--;
    }
    
    PowerSpectra* ps = PowerSpectra_alloc(nfft,wt);
    if(!ps) {
        WARN(ALLOCFAILED "ps");
        feTRACE(15);
        return NULL;
    }

    AvPowerSpectra* aps = a_malloc(sizeof(AvPowerSpectra));
    
    aps->nchannels = nchannels;
    aps->nfft = nfft;
    aps->ps = ps;
    aps->naverages = 0;
    aps->overlap = overlap;

    aps->buffer = dmat_alloc(nfft,nchannels);

    /* Allocate vectors and matrices */
    aps->ps_storage = cmat_alloc(nfft/2+1,nchannels*nchannels);
    aps->ps_single = cmat_alloc(nfft/2+1,nchannels*nchannels);
    aps->fifo = dFifo_create(nchannels,FIFO_SIZE_MULT*nfft);
    
    cmat_set(&aps->ps_storage,0);
    feTRACE(15);
    return aps;
}


us AvPowerSpectra_getAverages(const AvPowerSpectra* ps) {
    return ps->naverages;
}

/** 
 * Helper function that adds a block of time data to the APS
 *
 * @param aps AvPowerSpectra handle 
 * @param block Time data block. Size should be exactly nfft*nchannels.
 */
static void AvPowerSpectra_addBlock(AvPowerSpectra* aps,
                                    const dmat* block) {
    fsTRACE(15);
    
    dbgassert(aps && block,NULLPTRDEREF);
    dbgassert(block->n_rows == aps->nfft,"Invalid block n_rows");
    dbgassert(block->n_cols == aps->nchannels,"Invalid block n_cols");

    const us nfft = aps->nfft;
    iVARTRACE(15,nfft);
    
    cmat* ps_single = &aps->ps_single;
    cmat* ps_storage = &aps->ps_storage;
    
    c naverages = (++aps->naverages);

    /* Scale previous result */
    cmat_scale(ps_storage,
               (naverages-1)/naverages);

    
    PowerSpectra_compute(aps->ps,
                         block,
                         ps_single);


    /* Add new result, scaled properly */
    cmat_add_cmat(ps_storage,
                  ps_single,1/naverages);
    
    feTRACE(15);
}


cmat* AvPowerSpectra_addTimeData(AvPowerSpectra* aps,
                                 const dmat* timedata) {
    
    fsTRACE(15);
    
    dbgassert(aps && timedata,NULLPTRDEREF);
    const us nchannels = aps->nchannels;
    const us nfft = aps->nfft;
    
    dbgassert(timedata->n_cols == nchannels,"Invalid time data");
    dbgassert(timedata->n_rows > 0,"Invalid time data. "
              "Should at least have one row");

    const us nsamples = timedata->n_rows;

    /* Split up timedata in blocks of size ~ (FIFO_SIZE_MULT-1)nfft */
    const us max_blocksize = (FIFO_SIZE_MULT-1)*nfft;

    us pos = 0; /* Current position in timedata buffer */

    /* dFifo handle */
    dFifo* fifo = aps->fifo;

    do {
        us nsamples_part = pos+max_blocksize <= nsamples ?
            max_blocksize : nsamples-pos;

        /* Obtain sub matrix */
        dmat timedata_part = dmat_submat(timedata,
                                         pos, /* Startrow */
                                         0,   /* Startcol */
                                         nsamples_part, /* n_rows */
                                         nchannels);    /* n_cols */
    
        if(dFifo_push(fifo,&timedata_part)!=SUCCESS) {
            WARN("Fifo push failed.");
        }

        /* Temporary storage buffer */
        dmat* buffer = &aps->buffer;

        /* Pop samples from the fifo while there are still at
         * least nfft samples available */
        while (dFifo_size(fifo) >= nfft) {
            int popped = dFifo_pop(fifo,
                                   buffer,
                                   aps->overlap); /* Keep 'overlap'
                                                   * number of samples
                                                   * in the queue */

            dbgassert((us) popped == nfft,"Bug in dFifo");
            /* Process the block of time data */
            AvPowerSpectra_addBlock(aps,buffer);

        }

        dmat_free(&timedata_part);

        /* Update position */
        pos+=nsamples_part;
        
    } while (pos < nsamples);
    dbgassert(pos == nsamples,"BUG");
    
    feTRACE(15);
    return &aps->ps_storage;
}




//////////////////////////////////////////////////////////////////////
