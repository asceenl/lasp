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
    us nchannels;
    vd fft_work;
} Fft;

Fft* Fft_alloc(const us nfft,const us nchannels) {
    fsTRACE(15);

    if(nchannels > ASCEE_MAX_NUM_CHANNELS) {
        WARN("Too high number of channels! Please increase the "
             "ASCEE_MAX_NUM_CHANNELS compilation flag");
        return NULL;
    }

    Fft* fft = a_malloc(sizeof(Fft));
    if(fft==NULL) {
        WARN("Fft allocation failed");
        return NULL;
    }

    fft->nfft = nfft;
    fft->nchannels = nchannels;

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
us Fft_nchannels(const Fft* fft) {return fft->nchannels;}
us Fft_nfft(const Fft* fft) {return fft->nfft;}
void Fft_fft(const Fft* fft,const dmat* timedata,cmat* result) {
    fsTRACE(15);
    
    dbgassert(fft && timedata && result,NULLPTRDEREF);
    dbgassert(timedata->n_rows == fft->nfft,"Invalid size for time data rows."
        " Should be equal to nfft");
    dbgassert(timedata->n_cols == fft->nchannels,"Invalid size for time data cols."
        " Should be equal to nchannels");

    const us nfft = fft->nfft;

    vd fft_result = vd_alloc(fft->nfft);
    for(us col=0;col<fft->nchannels;col++) {

        vd timedata_col = dmat_column((dmat*) timedata,col);
        vd_copy(&fft_result,&timedata_col);
        vd_free(&timedata_col);

        npy_rfftf(fft->nfft,fft_result.ptr,fft->fft_work.ptr);

        /* Fftpack stores the data a bit strange, the resulting array
         * has the DC value at 0,the first cosine at 1, the first sine
         * at 2 etc. This needs to be shifted properly in the
         * resulting matrix, as for the complex data, the imaginary
         * part of the DC component equals zero. */

        check_overflow_vx(fft_result);
        check_overflow_vx(fft->fft_work);

        vc resultcol = cmat_column(result,col);
        *getvcval(&resultcol,0) = *getvdval(&fft_result,0);

        memcpy((void*) getvcval(&resultcol,1),
               (void*) getvdval(&fft_result,1),
               (nfft-1)*sizeof(d));

        /* Set imaginary part of Nyquist frequency to zero */
        ((d*) getvcval(&resultcol,nfft/2))[1] = 0;
        check_overflow_xmat(*timedata);

        /* Free up storage of the result column */
        vc_free(&resultcol);
    }

    /* print_vd(&fft->fft_work); */
    
    vd_free(&fft_result);
    feTRACE(15);
}

//////////////////////////////////////////////////////////////////////
