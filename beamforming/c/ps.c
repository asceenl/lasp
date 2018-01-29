// ps.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS 1000
#include "ps.h"
#include "fft.h"
#include "ascee_alloc.h"
#include "ascee_math.h"
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
    
    TRACE(15,"PowerSpectra_alloc");
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
    return ps;
}

void PowerSpectra_free(PowerSpectra* ps) {
    TRACE(15,"PowerSpectra_free");

    Fft_free(ps->fft);
    vd_free(&ps->window);
    cmat_free(&ps->fft_work);
    dmat_free(&ps->timedata_work);
    vc_free(&ps->j_vec_conj);
    a_free(ps);
}


int PowerSpectra_compute(const PowerSpectra* ps,
                         const dmat * timedata,
                         cmat * result) {
    
    TRACE(15,"PowerSpectra_compute");

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

    
    /* Scale fft such that power is easily comxputed */
    const c scale_fac = d_sqrt(2/win_pow)/nfft;
    c_scale(fft_work.data,scale_fac,(nfft/2+1)*nchannels);

    for(i=0;i< nchannels;i++) {
        /* Multiply DC term with 1/sqrt(2) */
        *getcmatval(&fft_work,0,i) *= 1/d_sqrt(2.)+0*I;

        /* Multiply Nyquist term with 1/sqrt(2) */
        *getcmatval(&fft_work,nfft/2,i) *= 1/d_sqrt(2.)+0*I;
    }

    /* print_cmat(&fft_work); */

    c* j_vec_conj = ps->j_vec_conj.data;

    /* Compute Cross-power spectra and store result */
    for(i =0; i<nchannels;i++) {
        for(j=0;j<nchannels;j++) {
            /* The indices here are important. This is also how it
             * is documented */
            c* res = getcmatval(result,0,i+j*nchannels);

            c* i_vec = getcmatval(&fft_work,0,i);
            c* j_vec = getcmatval(&fft_work,0,j);

            /* Compute the conjugate of spectra j */
            c_conj_c(j_vec_conj,j_vec,nfft/2+1);

            /* Compute the product of the two vectors and store the
             * result as the result */
            c_elem_prod_c(res,i_vec,j_vec_conj,nfft/2+1);

        }
    }

    return SUCCESS;
}




/* typedef struct AvPowerSpectra_s { */


/*     us overlap_offset; */
/*     us naverages;               /\* The number of averages taken *\/ */
    
/*     dmat prev_timedata;         /\**< Storage for previous */
/*                                  * timedata buffer *\/ */

/*     vc* ps;           /\**< Here we store the averaged */
/*                        * results for each Cross-power */
/*                        * spectra. These are */
/*                        * nchannels*nchannels vectors *\/ */

/*     vc* ps_work;         /\**< Work area for the results, also */
/*                           * nchannels*nchannels *\/ */

/* } AvPowerSpectra; */

/* AvPowerSpectra* AvPowerSpectra_alloc(const us nfft, */
/*                                      const us nchannels, */
/*                                      const d overlap_percentage) { */
    
/*     TRACE(15,"AvPowerSpectra_alloc"); */
/*     int rv; */

/*     /\* Check nfft *\/ */
/*     if(nfft % 2 != 0) { */
/*         WARN("nfft should be even"); */
/*         return NULL; */
/*     } */

/*     /\* Check overlap percentage *\/ */
/*     us overlap_offset = nfft - (us) overlap_percentage*nfft/100; */

/*     if(overlap_offset == 0 || overlap_offset > nfft) { */

/*         WARN("Overlap percentage results in offset of 0, or a too high number of */
/* overlap" */
/*              " samples. Number of overlap samples should be < nfft"); */

/*         WARN("Illegal overlap percentage. Should be 0 <= %% < 100"); */
/*         return NULL; */
/*     } */


/*     /\* ALlocate space *\/ */
/*     Fft fft; */
/*     rv = Fft_alloc(&fft,nfft,nchannels); */
/*     if(rv != SUCCESS) { */
/*         WARN("Fft allocation failed"); */
/*         return NULL; */
/*     } */

/*     AvPowerSpectra* aps = a_malloc(sizeof(AvPowerSpectra)); */
/*     if(!aps) { */
/*         WARN("Allocation of AvPowerSpectra memory failed"); */
/*         return NULL; */
/*     } */
/*     ps->naverages = 0; */
/*     ps->overlap_offset = overlap_offset; */

/*     /\* Allocate vectors and matrices *\/ */
/*     ps->prev_timedata = dmat_alloc(nfft,nchannels); */

/*     return ps; */
/* } */
/* us AvPowerSpectra_getAverages(const AvPowerSpectra* ps) { */
/*     return ps->naverages; */
/* } */

/* /\**  */
/*  * Compute single power spectra by  */
/*  * */
/*  * @param ps Initialized AvPowerSpectra structure */
/*  * @param timedata Timedata to compute for */
/*  * @param result Result */
/*  * */
/*  * @return  */
/*  *\/ */
/* static int AvPowerSpectra_computeSingle(const AvPowerSpectra* ps, */
/*                                       const dmat* timedata, */
/*                                       vc* results) { */

/*     us nchannels = ps->fft.nchannels; */
/*     for(us channel=0;channel<ps;channel++) { */

/*     } */
/*     return SUCCESS; */
/* } */


/* int AvPowerSpectra_addTimeData(AvPowerSpectra* ps, */
/*                              const dmat* timedata) { */
    
/*     TRACE(15,"AvPowerSpectra_addTimeData"); */
/*     dbgassert(ps,"Null pointer given for ps"); */
    
/*     const us nchannels = ps->fft.nchannels; */
/*     const us nfft = ps->fft.nfft; */

/*     dbgassert(timedata->n_cols == nchannels,"Invalid time data"); */
/*     dbgassert(timedata->n_rows == nfft,"Invalid time data"); */

/*     dmat prevt = ps->prev_timedata; */
/*     us noverlap = ps->noverlap; */

/*     if(ps->naverages != 0) { */
/*         /\* Copy the overlap buffer to the tbuf *\/ */
/*         copy_dmat_rows(&tbuf,&overlap,0,0,noverlap); */

/*         /\* Copy the new timedata buffer to the tbuf *\/ */
/*         copy_dmat_rows(&tbuf,timedata,0,noverlap,(nfft-noverlap)); */
/*     } */
/*     else { */
/*         /\* Copy the overlap buffer to the tbuf *\/ */
/*         copy_dmat_rows(&tbuf,&overlap,0,0,noverlap); */


/*     } */

/*     return SUCCESS; */
/* } */
/* void AvPowerSpectra_free(AvPowerSpectra* ps) { */
/*     TRACE(15,"AvPowerSpectra_free"); */

/*     Fft_free(&ps->fft); */
/*     dmat_free(&ps->prev_timedata); */
/*     vd_free(&ps->window);     */

/*     a_free(ps); */
/* } */

//////////////////////////////////////////////////////////////////////
