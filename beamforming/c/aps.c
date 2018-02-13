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

typedef struct AvPowerSpectra_s {
    us os;              /**< Counter set to the position where
                         * the next time block should be taken
                         * from */
    us nfft, nchannels;
    us oo;

    us naverages;               /* Counter that counts the number of
                                 * averages taken for the computation
                                 * of this averaged power spectra. */
    
    dmat buffer;                /**< Buffer storage of some of the
                                 * previous samples. Number of rows is
                                 * equal to nfft. */


    cmat ps_storage;           /**< Here we store the averaged
                                * results for each Cross-power
                                * spectra computed so far. */
    
    cmat ps_single;             /**< This is the work area for a
                                 * PowerSpectra computation on a
                                 * single block */

    PowerSpectra* ps;           /**< Pointer to underlying
                                 * PowerSpectra calculator. */
    
} AvPowerSpectra;

AvPowerSpectra* AvPowerSpectra_alloc(const us nfft,
                                     const us nchannels,
                                     const d overlap_percentage,
                                     const WindowType wt) {
    
    fsTRACE(15);
    
    /* Check nfft */
    if(nfft % 2 != 0 || nfft > ASCEE_MAX_NFFT) {
        WARN("nfft should be even");
        feTRACE(15);
        return NULL;
    }

    /* Check overlap percentage */
    if(overlap_percentage >= 100) {
        WARN("Overlap percentage >= 100!");
        feTRACE(15);
        return NULL;
    }
    if(overlap_percentage < 0) {
        WARN("Overlap percentage should be positive!");
        feTRACE(15);
        return NULL;
    }
    
    /* Compute and check overlap offset */
    us oo = (us) (((d) nfft)-overlap_percentage*((d) nfft)/100);
    iVARTRACE(15,oo);
    if(oo == 0) {oo++;}
        
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
    aps->oo = oo;
    aps->os = oo;

    /* Allocate vectors and matrices */
    aps->buffer = dmat_alloc(nfft,nchannels);
    aps->ps_storage = cmat_alloc(nfft/2+1,nchannels*nchannels);
    aps->ps_single = cmat_alloc(nfft/2+1,nchannels*nchannels);

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
    dbgassert(timedata->n_rows >= nfft,"Invalid time data. "
              "Should at least have nfft rows");

    const us oo = aps->oo;
    us* os = &aps->os;

    us os_timedata = 0;

    dmat buffer = aps->buffer;

    /* Retrieve the buffer and use it to make the first time block. */
    if(*os < oo) {
        TRACE(15,"Using saved data from previous run");
        dbgassert(false,"not tested")
        dmat tmp = dmat_alloc(nfft,nchannels);
        dbgassert(0 <= *os,"BUG");
        dbgassert(*os <= nfft,"BUG"); 

        /* copy_dmat_rows(&tmp, */
        /*                &buffer, */
        /*                *os,       /\* Startrow_from *\/ */
        /*                0,         /\* Startrow to *\/ */
        /*                nfft - *os /\* nrows *\/ */
        /*     ); */

        /* copy_dmat_rows(&tmp, */
        /*                timedata, */
        /*                0, */
        /*                nfft - *os, */
        /*                *os */
        /*     ); */


        AvPowerSpectra_addBlock(aps,&tmp);

        os_timedata = oo + *os - nfft;
        dbgassert(os_timedata < nfft,"BUG");
        dmat_free(&tmp);
    }

    /* Run until we cannot go any further */
    while ((os_timedata + nfft) <= timedata->n_rows) {

        dmat tmp = dmat_submat(timedata,
                               os_timedata, /* Startrow */
                               0,           /* Start column */
                               nfft,        /* Number of rows */
                               nchannels);  /* Number of columns */

        /* Process the block of time data */
        AvPowerSpectra_addBlock(aps,&tmp);

        iVARTRACE(15,os_timedata);
        os_timedata += oo;

        dmat_free(&tmp);
        iVARTRACE(15,os_timedata);
    }

    /* We copy the last piece of samples from the timedata to the
     * buffer */
    dmat_copy_rows(&buffer,
                   timedata,
                   0,           /* startrow_to */
                   timedata->n_rows-nfft, /* startrow_from */
                   nfft);       /* Number of rows */
    
    *os = os_timedata+nfft-timedata->n_rows;
    
    dbgassert(*os <= nfft,"BUG");

    feTRACE(15);
    return &aps->ps_storage;
}

void AvPowerSpectra_free(AvPowerSpectra* aps) {
    fsTRACE(15);

    PowerSpectra_free(aps->ps);
    dmat_free(&aps->buffer);
    cmat_free(&aps->ps_storage);
    cmat_free(&aps->ps_single);
    a_free(aps);
    
    feTRACE(15);
}


//////////////////////////////////////////////////////////////////////
