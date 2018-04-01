// lasp_aps.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-5)
#include "lasp_aps.h"
#include "lasp_ps.h"
#include "lasp_alg.h"
#include "lasp_dfifo.h"

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

    cmat* ps_storage;           /**< Here we store the averaged
                                 * results for each Cross-power
                                 * spectra computed so far. */

    cmat ps_result;

    cmat ps_single;             /**< This is the work area for a
                                 * PowerSpectra computation on a
                                 * single block */

    
    vd weighting;   /**< This array stores the time weighting
                     * coefficients for a running
                     * spectrogram. The vector length is
                     * zero for a full averaged (not
                     * running spectrogram). */
    
    us oldest_block;            /**< Index of oldest block in
                                 * Spectrogram mode */
    

    PowerSpectra* ps;           /**< Pointer to underlying
                                 * PowerSpectra calculator. */
    
} AvPowerSpectra;

void AvPowerSpectra_free(AvPowerSpectra* aps) {
    fsTRACE(15);

    PowerSpectra_free(aps->ps);
    dFifo_free(aps->fifo);
    dmat_free(&aps->buffer);

    us nweight = aps->weighting.size;
    if(nweight > 0) {
        for(us blockno = 0; blockno < nweight; blockno++) {
            cmat_free(&aps->ps_storage[blockno]);
        }
        a_free(aps->ps_storage);
        vd_free(&aps->weighting);
    }

    cmat_free(&aps->ps_single);
    cmat_free(&aps->ps_result);
    a_free(aps);
    
    feTRACE(15);
}

AvPowerSpectra* AvPowerSpectra_alloc(const us nfft,
                                     const us nchannels,
                                     const d overlap_percentage,
                                     const WindowType wt,
                                     const vd* weighting) {
    
    fsTRACE(15);
    
    /* Check nfft */
    if(nfft==0 || nfft % 2 != 0 || nfft > LASP_MAX_NFFT) {
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
    aps->oldest_block = 0;
    if(weighting) {
        
        us nweight = weighting->size;
        iVARTRACE(15,nweight);
        /* Allocate vectors and matrices */
        aps->ps_storage = a_malloc(nweight*sizeof(cmat));
        for(us blockno = 0; blockno < nweight; blockno++) {
            aps->ps_storage[blockno] = cmat_alloc(nfft/2+1,nchannels*nchannels);
            cmat_set(&aps->ps_storage[blockno],0);
        }

        /* Allocate space and copy weighting coefficients */
        aps->weighting = vd_alloc(weighting->size);
        vd_copy(&aps->weighting,weighting);
    }
    else {
        TRACE(15,"no weighting");
        aps->weighting.size = 0;
    }

    aps->ps_result = cmat_alloc(nfft/2+1,nchannels*nchannels);    
    aps->ps_single = cmat_alloc(nfft/2+1,nchannels*nchannels);
    cmat_set(&aps->ps_result,0);

    aps->fifo = dFifo_create(nchannels,FIFO_SIZE_MULT*nfft);

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
    cmat* ps_storage = aps->ps_storage;
    cmat* ps_result = &aps->ps_result;
    
    PowerSpectra_compute(aps->ps,
                         block,
                         ps_single);
    
    vd weighting = aps->weighting;
    us nweight = weighting.size;
    
    if(nweight == 0) {

        /* Overall mode */
        c naverages = (++aps->naverages);
    
        /* Scale previous result */
        cmat_scale(ps_result,
                   (naverages-1)/naverages);

        /* Add new result, scaled properly */
        cmat_add_cmat(ps_result,
                      ps_single,1/naverages);

    }
    else {
        cmat_set(ps_result,0);
        
        us* oldest_block = &aps->oldest_block;
        /* uVARTRACE(20,*oldest_block); */
        
        cmat_copy(&ps_storage[*oldest_block],ps_single);
        
        uVARTRACE(16,*oldest_block);
        if(aps->naverages < nweight) {
            ++(aps->naverages);
        }
        
        /* Update pointer to oldest block */
        (*oldest_block)++;
        *oldest_block %= nweight;

        us block_index_weight = *oldest_block;
        
        for(us block = 0; block < aps->naverages; block++) {
            /* Add new result, scaled properly */

            c weight_fac = *getvdval(&weighting,
                                     aps->naverages-1-block);

            /* cVARTRACE(20,weight_fac); */
            cmat_add_cmat(ps_result,
                          &ps_storage[block_index_weight],
                          weight_fac);

            block_index_weight++;
            block_index_weight %= nweight;


        }



    }
    
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

    /* dFifo handle */
    dFifo* fifo = aps->fifo;
    dFifo_push(fifo,timedata);

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
    
    feTRACE(15);
    return &aps->ps_result;
}




//////////////////////////////////////////////////////////////////////
