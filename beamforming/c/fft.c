// fft.cpp
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// 
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-5)
#include "ascee_tracer.h"
#include "fft.h"
#include "types.h"
#include "fftpack.h"

typedef struct Fft_s {
    us nfft;
    vd fft_work;
} Fft;

Fft* Fft_alloc(const us nfft) {
    fsTRACE(15);

    Fft* fft = a_malloc(sizeof(Fft));
    if(fft==NULL) {
        WARN("Fft allocation failed");
        return NULL;
    }

    fft->nfft = nfft;

    /* Initialize foreign fft lib */
    fft->fft_work = vd_alloc(2*nfft+15);
    npy_rffti(nfft,fft->fft_work.ptr);
    check_overflow_vx(fft->fft_work);

    /* print_vd(&fft->fft_work); */
    
    feTRACE(15); 
    return fft;
}
void Fft_free(Fft* fft) {
    fsTRACE(15);
    dbgassert(fft,NULLPTRDEREF);
    vd_free(&fft->fft_work);
    a_free(fft);
    feTRACE(15);
}
us Fft_nfft(const Fft* fft) {return fft->nfft;}
void Fft_fft_single(const Fft* fft,const vd* timedata,vc* result) {
    
    fsTRACE(15);
    dbgassert(fft && timedata && result,NULLPTRDEREF);

    const us nfft = fft->nfft;

    dbgassert(timedata->size == nfft,
              "Invalid size for time data rows."
              " Should be equal to nfft");

    dbgassert(result->size == (nfft/2+1),"Invalid number of rows in"
              " result array");

    vd fft_result = vd_alloc(nfft);

    vd_copy(&fft_result,timedata);

    /* Perform fft */
    npy_rfftf(nfft,fft_result.ptr,fft->fft_work.ptr);

    /* Fftpack stores the data a bit strange, the resulting array
     * has the DC value at 0,the first cosine at 1, the first sine
     * at 2 etc. This needs to be shifted properly in the
     * resulting matrix, as for the complex data, the imaginary
     * part of the DC component equals zero. */

    *getvcval(result,0) = *getvdval(&fft_result,0);

    memcpy((void*) getvcval(result,1),
           (void*) getvdval(&fft_result,1),
           (nfft-1)*sizeof(d));

    /* Set imaginary part of Nyquist frequency to zero */
    ((d*) getvcval(result,nfft/2))[1] = 0;

    check_overflow_vx(fft_result);
    check_overflow_vx(fft->fft_work);
    vd_free(&fft_result);
    feTRACE(15);

}
void Fft_fft(const Fft* fft,const dmat* timedata,cmat* result) {
    fsTRACE(15);
    
    dbgassert(fft && timedata && result,NULLPTRDEREF);

    const us nchannels = timedata->n_cols;
    dbgassert(timedata->n_cols == result->n_cols,
              "Number of columns in timedata and result"
              " should be equal.");

    for(us col=0;col<nchannels;col++) {

        vd timedata_col = dmat_column((dmat*) timedata,col);
        vc result_col = cmat_column(result,col);

        Fft_fft_single(fft,&timedata_col,&result_col);

        vd_free(&timedata_col);
        vc_free(&result_col);
    }
    check_overflow_xmat(*timedata);
    check_overflow_xmat(*result);    


    feTRACE(15);
}

//////////////////////////////////////////////////////////////////////
