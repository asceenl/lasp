// fft.cpp
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// 
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-5)
#include "tracer.h"
#include "fft.h"
#include "types.h"
/* #include "kiss_fftr.h" */
#include "fftpack.h"

typedef struct Fft_s {
    us nfft;
    us nchannels;

    Fftr* fftr;

} Fft;

Fft* Fft_alloc(const us nfft,const us nchannels) {
    fsTRACE(15);

    #ifdef ASCEE_DEBUG
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

    fft->fftr = Fftr_alloc(nfft);
    if(!fft->fftr) {
        WARN(ALLOCFAILED "fftr");
        return NULL;
    }
    #endif // ASCEE_PARALLEL

    feTRACE(15); 
    return fft;
}
void Fft_free(Fft* fft) {
    fsTRACE(15);
    Fftr_free(fft->fftr);
    a_free(fft);
    feTRACE(15);
}
us Fft_nchannels(const Fft* fft) {return fft->nchannels;}
us Fft_nfft(const Fft* fft) {return fft->nfft;}

void Fft_fft(const Fft* fft,const dmat* timedata,cmat* result) {
    fsTRACE(15);

    dbgassert(timedata->n_rows == fft->nfft,"Invalid size for time data rows."
        " Should be equal to nfft");
    dbgassert(timedata->n_cols == fft->nchannels,"Invalid size for time data cols."
        " Should be equal to nchannels");
    
    for(us col=0;col<fft->nchannels;col++) {

        Fftr_fftr(fft->fftr,
             getdmatval(timedata,0,col),
             getcmatval(result,0,col));
        
    }

    feTRACE(15);
}

//////////////////////////////////////////////////////////////////////
