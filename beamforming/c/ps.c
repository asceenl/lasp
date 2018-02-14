// ps.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-5)
#include "ps.h"
#include "fft.h"
#include "ascee_alloc.h"
#include "ascee_alg.h"
#include "ascee_assert.h"

typedef struct PowerSpectra_s {

    vd window;
    d win_pow;                  /**< The power of the window */
    Fft* fft;                    /**< Fft routines storage */
} PowerSpectra;

PowerSpectra* PowerSpectra_alloc(const us nfft,
                                 const WindowType wt) {
    
    fsTRACE(15);
    int rv;

    /* Check nfft */
    if(nfft % 2 != 0) {
        WARN("nfft should be even");
        return NULL;
    }

    /* ALlocate space */
    Fft* fft = Fft_alloc(nfft);
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
    
    rv = window_create(wt,&ps->window,&ps->win_pow);
    check_overflow_vx(ps->window);
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
    a_free(ps);
    feTRACE(15);
}


void PowerSpectra_compute(const PowerSpectra* ps,
                         const dmat * timedata,
                         cmat * result) {
    
    fsTRACE(15);

    dbgassert(ps && timedata && result,NULLPTRDEREF);
    
    const us nchannels = timedata->n_cols;
    const us nfft = Fft_nfft(ps->fft);
    uVARTRACE(15,nchannels);
    const d win_pow = ps->win_pow;
    dVARTRACE(15,win_pow);

    /* Sanity checks for the matrices */
    dbgassert(timedata->n_rows == nfft,"timedata n_rows "
              "should be equal to nfft");

    dbgassert(result->n_rows == nfft/2+1,"result n_rows "
              "should be equal to nfft/2+1");

    dbgassert(result->n_cols == nchannels*nchannels,"result n_cols "
              "should be equal to nchannels*nchannels");


    /* Multiply time data with the window and store result in
     * timedata_work. */
    dmat timedata_work = dmat_alloc(nfft,nchannels);
    for(us i=0;i<nchannels;i++) {
        vd column = dmat_column((dmat*) timedata,i);
        vd column_work = dmat_column(&timedata_work,i);

        vd_elem_prod(&column_work,&column,&ps->window);
        
        vd_free(&column);
        vd_free(&column_work);
    }
    check_overflow_xmat(timedata_work);
    /* print_dmat(&timedata_work); */
    
    /* Compute fft of the time data */
    cmat fft_work = cmat_alloc(nfft/2+1,nchannels);
    Fft_fft(ps->fft,
            &timedata_work,
            &fft_work);

    dmat_free(&timedata_work);
    
    TRACE(15,"fft done");
    
    /* Scale fft such that power is easily comxputed */
    const c scale_fac = d_sqrt(2/win_pow)/nfft;
    cmat_scale(&fft_work,scale_fac);
    TRACE(15,"scale done");

    for(us i=0;i< nchannels;i++) {
        /* Multiply DC term with 1/sqrt(2) */
        *getcmatval(&fft_work,0,i) *= 1/d_sqrt(2.)+0*I;

        /* Multiply Nyquist term with 1/sqrt(2) */
        *getcmatval(&fft_work,nfft/2,i) *= 1/d_sqrt(2.)+0*I;
    }
    check_overflow_xmat(fft_work);

    /* print_cmat(&fft_work); */
    TRACE(15,"Nyquist and DC correction done");

    vc j_vec_conj = vc_alloc(nfft/2+1);

    /* Compute Cross-power spectra and store result */
    for(us i =0; i<nchannels;i++) {
        for(us j=0;j<nchannels;j++) {

            /* The indices here are important. This is also how it
             * is documented */
            vc res = cmat_column(result,i+j*nchannels);

            vc i_vec = cmat_column(&fft_work,i);
            vc j_vec = cmat_column(&fft_work,j);

            /* Compute the conjugate of spectra j */
            vc_conj(&j_vec_conj,&j_vec);

            check_overflow_xmat(fft_work);

            /* Compute the element-wise product of the two vectors and
             * store the result as the result */
            vc_hadamard(&res,&i_vec,&j_vec_conj);

            vc_free(&i_vec);
            vc_free(&j_vec);
            vc_free(&res);

        }
    }
    check_overflow_xmat(*result);
    check_overflow_xmat(*timedata);
    cmat_free(&fft_work);
    vc_free(&j_vec_conj);
    feTRACE(15);
}



//////////////////////////////////////////////////////////////////////
