// ps.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
/* #define TRACERPLUS 1000 */
#include "ps.h"
#include "fft.h"
#include "ascee_alloc.h"
#include "ascee_alg.h"
#include "ascee_assert.h"

typedef struct PowerSpectra_s {

    vd window;

    d win_pow;                  /**< The power of the window */
    Fft* fft;                    /**< Fft routines storage */
    cmat fft_work;              /**< Work area for FFt's */
    dmat timedata_work;         /**< Work area for timedata */
    vc j_vec_conj;              /**< Work area for conjugate of j */
} PowerSpectra;

PowerSpectra* PowerSpectra_alloc(const us nfft,
                                 const us nchannels,
                                 const WindowType wt) {
    
    fsTRACE(15);
    int rv;

    /* Check nfft */
    if(nfft % 2 != 0) {
        WARN("nfft should be even");
        return NULL;
    }

    /* ALlocate space */
    Fft* fft = Fft_alloc(nfft,nchannels);
    if(fft == NULL) {
        WARN("Fft allocation failed");
        return NULL;
    }

    PowerSpectra* ps = a_malloc(sizeof(PowerSpectra));
    if(!ps) {
        WARN("Allocation of PowerSpectra memory failed");
        Fft_free(fft);
        return NULL;
    }
    ps->fft = fft;

    /* Allocate vectors and matrices */
    ps->window = vd_alloc(nfft);
    ps->fft_work = cmat_alloc(nfft/2+1,nchannels);
    ps->timedata_work= dmat_alloc(nfft,nchannels);
    ps->j_vec_conj = vc_alloc(nfft/2+1);
    
    rv = window_create(wt,&ps->window,&ps->win_pow);
    if(rv!=0) {
        WARN("Error creating window function, continuing anyway");
    }
    feTRACE(15);
    return ps;
}

void PowerSpectra_free(PowerSpectra* ps) {
    fsTRACE(15);

    Fft_free(ps->fft);
    vd_free(&ps->window);
    cmat_free(&ps->fft_work);
    dmat_free(&ps->timedata_work);
    vc_free(&ps->j_vec_conj);
    a_free(ps);

    feTRACE(15);
}


void PowerSpectra_compute(const PowerSpectra* ps,
                         const dmat * timedata,
                         cmat * result) {
    
    fsTRACE(15);

    dbgassert(ps && timedata && result,NULLPTRDEREF);
    
    const us nchannels = Fft_nchannels(ps->fft);
    const us nfft = Fft_nfft(ps->fft);
    uVARTRACE(15,nchannels);
    const d win_pow = ps->win_pow;
    dVARTRACE(15,win_pow);

    us i,j;

    /* Sanity checks for the matrices */
    dbgassert(timedata->n_cols == nchannels,"timedata n_cols "
              "should be equal to nchannels");

    dbgassert(timedata->n_rows == nfft,"timedata n_rows "
              "should be equal to nfft");

    dbgassert(result->n_rows == nfft/2+1,"result n_rows "
              "should be equal to nfft/2+1");

    dbgassert(result->n_cols == nchannels*nchannels,"result n_cols "
              "should be equal to nchannels*nchannels");


    /* Multiply time data with the window and store result in
     * timedata_work. */
    dmat timedata_work = ps->timedata_work;

    for(i=0;i<nchannels;i++) {

        d_elem_prod_d(getdmatval(&timedata_work,0,i), /* Result */
                      getdmatval(timedata,0,i),
                      ps->window.data,
                      nfft);
                                 
    }

    /* print_dmat(&timedata_work); */
    
    /* Compute fft of the time data */
    cmat fft_work = ps->fft_work;
    Fft_fft(ps->fft,
            &timedata_work,
            &fft_work);

    TRACE(15,"fft done");
    
    /* Scale fft such that power is easily comxputed */
    const c scale_fac = d_sqrt(2/win_pow)/nfft;
    cmat_scale(&fft_work,scale_fac);
    TRACE(15,"scale done");
    
    for(i=0;i< nchannels;i++) {
        /* Multiply DC term with 1/sqrt(2) */
        *getcmatval(&fft_work,0,i) *= 1/d_sqrt(2.)+0*I;

        /* Multiply Nyquist term with 1/sqrt(2) */
        *getcmatval(&fft_work,nfft/2,i) *= 1/d_sqrt(2.)+0*I;
    }

    /* print_cmat(&fft_work); */
    TRACE(15,"Nyquist and DC correction done");

    vc j_vec_conj = ps->j_vec_conj;

    /* Compute Cross-power spectra and store result */
    for(i =0; i<nchannels;i++) {
        for(j=0;j<nchannels;j++) {
            iVARTRACE(15,i);
            iVARTRACE(15,j);
            /* The indices here are important. This is also how it
             * is documented */
            vc res = cmat_column(result,i+j*nchannels);
            TRACE(15,"SFSG");
            vc i_vec = cmat_column(&fft_work,i);
            vc j_vec = cmat_column(&fft_work,j);
            TRACE(15,"SFSG");
            /* Compute the conjugate of spectra j */
            vc_conj_vc(&j_vec_conj,&j_vec);
            TRACE(15,"SFSG");
            /* Compute the element-wise product of the two vectors and
             * store the result as the result */
            vc_elem_prod(&res,&i_vec,&j_vec_conj);
            TRACE(15,"SFSG");
        }
    }
    feTRACE(15);
}



//////////////////////////////////////////////////////////////////////
