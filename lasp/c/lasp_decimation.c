// lasp_decimation.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// Implementation of the decimator
//////////////////////////////////////////////////////////////////////
#include "lasp_decimation.h"
#include "lasp_firfilterbank.h"
#include "lasp_tracer.h"
#include "lasp_alloc.h"
#include "lasp_dfifo.h"

// The Maximum number of taps in a decimation filter
#define DEC_FILTER_MAX_TAPS (128)

// The FFT length
#define DEC_FFT_LEN (1024)

typedef struct {
    DEC_FAC df;
    us dec_fac;
    us ntaps;
    d h[DEC_FILTER_MAX_TAPS];
} DecFilter;

static __thread DecFilter DecFilters[] = {
    {DEC_FAC_4,4,128,
#include "dec_filter_4.c"
    }
};


typedef struct Decimator_s {
    us nchannels;
    us dec_fac;
    Firfilterbank** fbs;
    dFifo* output_fifo;
} Decimator;


Decimator* Decimator_create(us nchannels,DEC_FAC df) {
    fsTRACE(15);

    /* Find the filter */
    const us nfilters = sizeof(DecFilters)/sizeof(DecFilter);
    DecFilter* filter = DecFilters;
    bool found = false;
    for(us filterno = 0;filterno < nfilters; filterno++) {
        if(filter->df == df) {
            TRACE(15,"Found filter");
            found = true;
            break;
        }
        filter++;
    }
    if(!found) {
        WARN("Decimation factor not found in list of filters");
        return NULL;
    }
    
    /* Create the filterbanks */
    Decimator* dec = a_malloc(sizeof(Decimator));
    dec->fbs = a_malloc(sizeof(Firfilterbank*)*nchannels);
    dec->nchannels = nchannels;
    dec->dec_fac = filter->dec_fac;

    dmat h = dmat_foreign_data(filter->ntaps,1,filter->h,false);

    for(us channelno=0;channelno<nchannels;channelno++) {
        dec->fbs[channelno] = Firfilterbank_create(&h,DEC_FFT_LEN);
    }

    dmat_free(&h);
    
    /* Create input and output fifo's */
    dec->output_fifo = dFifo_create(nchannels,DEC_FFT_LEN);

    feTRACE(15);
    return dec; 
}

static void lasp_downsample(dmat* downsampled,
                            const dmat* filtered,
                            us dec_fac) {
    fsTRACE(15);
    dbgassert(downsampled && filtered,NULLPTRDEREF);
    dbgassert(filtered->n_rows/dec_fac == downsampled->n_rows,
              "Incompatible number of rows");
    dbgassert(downsampled->n_cols == filtered->n_cols,
              "Incompatible number of rows");
    
    dbgassert(dec_fac> 0,"Invalid dec_fac");

    for(us col=0;col<downsampled->n_cols;col++) {
        /* Low-level BLAS copy. */
        d_copy(getdmatval(downsampled,0,col),
               getdmatval(filtered,0,col),
               downsampled->n_rows,     /* number */
               1,                       /* incx out */
               dec_fac);                /* incx in */
    }

    check_overflow_xmat(*downsampled);
    check_overflow_xmat(*filtered);

    feTRACE(15);
}

dmat Decimator_decimate(Decimator* dec,const dmat* samples) {

    fsTRACE(15);
    dbgassert(dec && samples,NULLPTRDEREF);
    const us nchannels = dec->nchannels;
    const us dec_fac = dec->dec_fac;
    dbgassert(samples->n_cols == nchannels,"Invalid number of "
              "channels in samples");
    dbgassert(samples->n_rows > 0,"Number of rows should be >0")

    /* Not downsampled, but filtered result */
    dmat filtered;

    /* Filter each channel and store result in filtered. In first
     * iteration the right size for filtered is allocated. */

    for(us chan=0;chan<nchannels;chan++) {
        vd samples_channel = dmat_column((dmat*)samples,
                                         chan);

        /* Low-pass filter stuff */
        dmat filtered_res = Firfilterbank_filter(dec->fbs[chan],
                                                 &samples_channel);

        dbgassert(filtered_res.n_cols == 1,"Bug in Firfilterbank");

        vd_free(&samples_channel);

        if(filtered_res.n_rows > 0) {
            if(chan==0) {
                /* Allocate space for result */
                filtered = dmat_alloc(filtered_res.n_rows,
                                      nchannels);

            }
            dmat filtered_col = dmat_submat(&filtered,
                                            0,chan,
                                            filtered_res.n_rows,
                                            1);

            dbgassert(filtered_res.n_rows == filtered_col.n_rows,
                      "Not all Firfilterbank's have same output number"
                      " of rows!");

            dmat_copy_rows(&filtered_col,
                           &filtered_res,
                           0,0,filtered_res.n_rows);

            dmat_free(&filtered_res);
            dmat_free(&filtered_col);
        }
        else {
            filtered = dmat_alloc(0, nchannels);
        }
    }
    if(filtered.n_rows > 0) {
        /* Push filtered result into output fifo */
        dFifo_push(dec->output_fifo,
                   &filtered);
    }
    dmat_free(&filtered);


    /* Now, downsample stuff */
    dmat downsampled;
    uVARTRACE(15,dec_fac);
    us fifo_size = dFifo_size(dec->output_fifo);
    if((fifo_size / dec_fac) > 0) {

        filtered = dmat_alloc((fifo_size/dec_fac)*dec_fac,
                              nchannels);

        int nsamples = dFifo_pop(dec->output_fifo,
                                 &filtered,0);

        dbgassert((us) nsamples % dec_fac == 0 && nsamples > 0,
                  "BUG");
        dbgassert((us) nsamples == (fifo_size/dec_fac)*dec_fac,"BUG");
        
        downsampled = dmat_alloc(nsamples/dec_fac,nchannels);
        /* Do the downsampling work */
        lasp_downsample(&downsampled,&filtered,dec_fac);
        
        dmat_free(&filtered);
    }
    else {
        TRACE(15,"Empty downsampled");
        downsampled = dmat_alloc(0,0);
    }
    
    feTRACE(15);
    /* return filtered; */
    return downsampled;
}


void Decimator_free(Decimator* dec) {
    fsTRACE(15);
    dbgassert(dec,NULLPTRDEREF);
    dFifo_free(dec->output_fifo);

    for(us chan=0;chan<dec->nchannels;chan++) {
        Firfilterbank_free(dec->fbs[chan]);
    }

    a_free(dec->fbs);
    a_free(dec);

    feTRACE(15);
}
//////////////////////////////////////////////////////////////////////

