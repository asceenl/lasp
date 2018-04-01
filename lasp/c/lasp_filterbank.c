// lasp_filterbank.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// FilterBank implementation.
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-5)
#include "lasp_filterbank.h"
#include "lasp_fft.h"
#include "lasp_dfifo.h"
#include "lasp_tracer.h"
#include "lasp_alg.h"
#define FIFO_SIZE_MULT 2

typedef struct FilterBank_s {
    us nfft;

    us P_m_1;                   /**< Filter length minus one */

    cmat filters;               /* Frequency-domain filter
                                 * coefficients */
    dFifo* output_fifo;
    dFifo* input_fifo;

    Fft* fft;                   /* Handle to internal FFT-function */

} FilterBank;

FilterBank* FilterBank_create(const dmat* h,
                              const us nfft) {

    fsTRACE(15);
    dbgassert(h,NULLPTRDEREF);
    const us P = h->n_rows;
    const us nfilters = h->n_cols;
    
    if(P > nfft/2) {
        WARN("Filter order should be <= nfft/2");
        return NULL;
    }

    Fft* fft = Fft_create(nfft);
    if(!fft) {
        WARN("Fft allocation failed");
        return NULL;
    }

    FilterBank* fb = a_malloc(sizeof(FilterBank));

    fb->nfft = nfft;
    fb->P_m_1 = P-1;
    fb->fft = fft;
    fb->filters = cmat_alloc(nfft/2+1,nfilters);

    fb->output_fifo = dFifo_create(nfilters,FIFO_SIZE_MULT*nfft);
    fb->input_fifo = dFifo_create(1,FIFO_SIZE_MULT*nfft);

    /* Create a temporary buffer which is going to be FFT'th to
     * contain the filter transfer functions.
     */
    dmat temp = dmat_alloc(nfft,nfilters);
    dmat_set(&temp,0);
    dmat_copy_rows(&temp,h,0,0,h->n_rows);

    /* Fft the FIR impulse responses */
    Fft_fft(fb->fft,&temp,&fb->filters);

    dmat_free(&temp);

    feTRACE(15);
    return fb;
}
void FilterBank_free(FilterBank* fb) {
    fsTRACE(15);
    dbgassert(fb,NULLPTRDEREF);
    cmat_free(&fb->filters);
    dFifo_free(fb->input_fifo);
    dFifo_free(fb->output_fifo);
    Fft_free(fb->fft);
    a_free(fb);
    feTRACE(15);
}
dmat FilterBank_filter(FilterBank* fb,
                       const vd* x) {

    fsTRACE(15);
    dbgassert(fb && x ,NULLPTRDEREF);

    dFifo* input_fifo = fb->input_fifo;
    dFifo* output_fifo = fb->output_fifo;
 
    const us nfft = fb->nfft;
    const us nfilters = fb->filters.n_cols;

    /* Push samples to the input fifo */
    dFifo_push(fb->input_fifo,x);

    dmat input_block = dmat_alloc(nfft,1);

    /* Output is ready to be multiplied with FFt'th filter */
    cmat input_fft = cmat_alloc(nfft/2+1,1);

    /* Output is ready to be multiplied with FFt'th filter */
    cmat output_fft = cmat_alloc(nfft/2+1,nfilters);
    dmat output_block = dmat_alloc(nfft,nfilters);

    while (dFifo_size(input_fifo) >= nfft) {

        us nsamples = dFifo_pop(input_fifo,
                                &input_block,
                                fb->P_m_1 /* save P-1 samples */
            );
        dbgassert(nsamples == nfft,"BUG in dFifo");

        Fft_fft(fb->fft,&input_block,&input_fft);

        vc input_fft_col = cmat_column(&input_fft,0);

        for(us col=0;col<nfilters;col++) {
            vc output_fft_col = cmat_column(&output_fft,col);
            vc filter_col = cmat_column(&fb->filters,col);

            vc_hadamard(&output_fft_col,
                        &input_fft_col,
                        &filter_col);
                        
            vc_free(&output_fft_col);
            vc_free(&filter_col);
        }
        vc_free(&input_fft_col);

        Fft_ifft(fb->fft,&output_fft,&output_block);

        dmat valid_samples = dmat_submat(&output_block,
                                         fb->P_m_1,0, /* startrow, startcol */
                                         nfft-fb->P_m_1, /* Number of rows */
                                         output_block.n_cols);
        dFifo_push(fb->output_fifo,&valid_samples);
        dmat_free(&valid_samples);

    }

    dmat_free(&input_block);
    cmat_free(&input_fft);
    cmat_free(&output_fft);
    dmat_free(&output_block);

    us samples_done = dFifo_size(output_fifo);
    uVARTRACE(15,samples_done);
    dmat filtered_result = dmat_alloc(samples_done,nfilters);
    if(samples_done) {
        us samples_done2 = dFifo_pop(output_fifo,&filtered_result,0);
        dbgassert(samples_done2 == samples_done,"BUG in dFifo");
    }
    feTRACE(15);
    return filtered_result;
}


//////////////////////////////////////////////////////////////////////
