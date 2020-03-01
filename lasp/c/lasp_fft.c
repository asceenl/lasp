// lasp_fft.c
//
// Author: J.A. de Jong - ASCEE
// 
// Description:
// 
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-5)
#include "lasp_tracer.h"
#include "lasp_fft.h"
#include "lasp_types.h"

#ifdef LASP_FFT_BACKEND_FFTPACK
#include "fftpack.h"
typedef struct Fft_s {
    us nfft;
    vd fft_work; // Storage memory for fftpack
};
#elif defined LASP_FFT_BACKEND_FFTW
#include <fftw3.h>
typedef struct Fft_s {
    us nfft;
    fftw_plan forward_plan;
    fftw_plan reverse_plan;
    c* complex_storage;
    d* real_storage;
};

#endif

void load_fft_wisdom(const char* wisdom) {
#ifdef LASP_FFT_BACKEND_FFTPACK
#elif defined LASP_FFT_BACKEND_FFTW
    if(wisdom) {
        int rv= fftw_import_wisdom_from_string(wisdom);
        if(rv != 1) {
            fprintf(stderr, "Error loading FFTW wisdom");
        }
    }
#endif
}

char* store_fft_wisdom() {
#ifdef LASP_FFT_BACKEND_FFTPACK
    return NULL;
#elif defined LASP_FFT_BACKEND_FFTW
    return fftw_export_wisdom_to_string();
#endif
}

Fft* Fft_create(const us nfft) {
    fsTRACE(15);
    if(nfft == 0) {
        WARN("nfft should be > 0");
        return NULL;
    }

    Fft* fft = a_malloc(sizeof(Fft));

    fft->nfft = nfft;

#ifdef LASP_FFT_BACKEND_FFTPACK
    /* Initialize foreign fft lib */
    fft->fft_work = vd_alloc(2*nfft+15);
    npy_rffti(nfft,getvdval(&fft->fft_work,0));
    check_overflow_vx(fft->fft_work);
#elif defined LASP_FFT_BACKEND_FFTW
    fft->complex_storage = fftw_malloc(sizeof(c) * (nfft/2 + 1));
    fft->real_storage = fftw_malloc(sizeof(d) * nfft);

    fft->forward_plan = fftw_plan_dft_r2c_1d(nfft,
            fft->real_storage,
            fft->complex_storage,
            FFTW_MEASURE);
    fft->reverse_plan = fftw_plan_dft_c2r_1d(nfft,
            fft->complex_storage,
            fft->real_storage,
            FFTW_MEASURE);

#endif

    /* print_vd(&fft->fft_work); */

    feTRACE(15); 
    return fft;
}
void Fft_free(Fft* fft) {
    fsTRACE(15);
    dbgassert(fft,NULLPTRDEREF);
#ifdef LASP_FFT_BACKEND_FFTPACK
    vd_free(&fft->fft_work);
#elif defined LASP_FFT_BACKEND_FFTW
    fftw_free(fft->complex_storage);
    fftw_free(fft->real_storage);
    fftw_destroy_plan(fft->forward_plan);
    fftw_destroy_plan(fft->reverse_plan);
#endif
    a_free(fft);
    feTRACE(15);
}

us Fft_nfft(const Fft* fft) {return fft->nfft;}

void Fft_ifft_single(const Fft* fft,const vc* freqdata,vd* result) {
    fsTRACE(15);
    dbgassert(fft && freqdata && result,NULLPTRDEREF);
    const us nfft = fft->nfft;
    dbgassert(result->n_rows == nfft,
            "Invalid size for time data rows."
            " Should be equal to nfft");

    dbgassert(freqdata->n_rows == (nfft/2+1),"Invalid number of rows in"
            " result array");


    d* result_ptr = getvdval(result,0);

#ifdef LASP_FFT_BACKEND_FFTPACK
    d* freqdata_ptr = (d*) getvcval(freqdata,0);
    /* Copy freqdata, to fft_result. */
    d_copy(&result_ptr[1],&freqdata_ptr[2],nfft-1,1,1);
    result_ptr[0] = freqdata_ptr[0];

    /* Perform inplace backward transform */
    npy_rfftb(nfft,
            result_ptr,
            getvdval(&fft->fft_work,0));


#elif defined LASP_FFT_BACKEND_FFTW
    c* freqdata_ptr = (c*) getvcval(freqdata,0);

    c_copy(fft->complex_storage, freqdata_ptr,nfft/2+1);

    fftw_execute(fft->reverse_plan);
    
    d_copy(result_ptr, fft->real_storage, nfft, 1, 1);
    
#endif
    check_overflow_vx(*result);

    /* Scale by dividing by nfft. Checked with numpy implementation
     * that this indeed needs to be done for FFTpack. */
    d_scale(result_ptr,1/((d) nfft),nfft);
    feTRACE(15);
}
void Fft_ifft(const Fft* fft,const cmat* freqdata,dmat* timedata) {
    fsTRACE(15);

    dbgassert(fft && timedata && freqdata,NULLPTRDEREF);

    const us nchannels = timedata->n_cols;
    dbgassert(timedata->n_cols == freqdata->n_cols,
            "Number of columns in timedata and result"
            " should be equal.");

    for(us col=0;col<nchannels;col++) {

        vd timedata_col = dmat_column(timedata,col);
        vc freqdata_col = cmat_column((cmat*)freqdata,col);

        Fft_ifft_single(fft,&freqdata_col,&timedata_col);

        vd_free(&timedata_col);
        vc_free(&freqdata_col);
    }
    check_overflow_xmat(*timedata);
    check_overflow_xmat(*freqdata);    

    feTRACE(15);
}

void Fft_fft_single(const Fft* fft,const vd* timedata,vc* result) {

    fsTRACE(15);
    dbgassert(fft && timedata && result,NULLPTRDEREF);

    const us nfft = fft->nfft;
    assert_vx(timedata);
    assert_vx(result);
    dbgassert(timedata->n_rows == nfft,
            "Invalid size for time data rows."
            " Should be equal to nfft");

    dbgassert(result->n_rows == (nfft/2+1),"Invalid number of rows in"
            " result array");


#ifdef LASP_FFT_BACKEND_FFTPACK
    d* result_ptr = (d*) getvcval(result,0);

    /* Fftpack stores the data a bit strange, the resulting array
     * has the DC value at 0,the first cosine at 1, the first sine
     * at 2 etc. 1
     * resulting matrix, as for the complex data, the imaginary
     * part of the DC component equals zero. */

    /* Copy timedata, as it will be overwritten in the fft pass. */
    d_copy(&result_ptr[1],getvdval(timedata,0),nfft,1,1);


    /* Perform fft */
    npy_rfftf(nfft,&result_ptr[1],
            getvdval(&fft->fft_work,0));

    /* Set real part of DC component to first index of the rfft
     * routine */
    result_ptr[0] = result_ptr[1];

    result_ptr[1] = 0;        /* Set imaginary part of DC component
                               * to zero */

    /* For an even fft, the imaginary part of the Nyquist frequency
     * bin equals zero.*/
    if(likely(nfft%2 == 0)) {
        result_ptr[nfft+1] = 0;
    }
    check_overflow_vx(fft->fft_work);
#elif defined LASP_FFT_BACKEND_FFTW

    d* timedata_ptr = getvdval(timedata,0);
    c* result_ptr = getvcval(result,0);
    d_copy(fft->real_storage,timedata_ptr, nfft, 1, 1);

    fftw_execute(fft->forward_plan);
    
    c_copy(result_ptr, fft->complex_storage, nfft/2+1);
    
#endif
    check_overflow_vx(*result);
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
