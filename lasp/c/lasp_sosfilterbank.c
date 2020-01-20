#define TRACERPLUS (-5)
#include "lasp_sosfilterbank.h"


typedef struct Sosfilterbank {
    /// The filter_coefs matrix contains filter coefficients for a SOS filter.
    us filterbank_size;
    us nsections;

    /// The filter coefficients for each of the filters in the Filterbank
    /// The *first* axis is the filter no, the second axis contains the
    /// filter coefficients, in the order, b_0, b_1, b_2, a_0, a_1, a_2, which
    /// corresponds to the transfer function 
    ///                 b_0 + b_1 z^-1 + b_2 z^-2
    ///         H[z] =  -------------------------
    ///                 a_0 + a_1 z^-1 + a_2 z^-2 
    dmat sos; /// sos[filter_no, coeff]

    /// Storage for the current state of the output, first axis correspond to
    /// the filter number axis, the second axis contains state coefficients 
    dmat state;
} Sosfilterbank;

us Sosfilterbank_getFilterbankSize(const Sosfilterbank* fb) {
    fsTRACE(15);
    assertvalidptr(fb);
    return fb->filterbank_size;
    feTRACE(15);

}

Sosfilterbank* Sosfilterbank_create(const us filterbank_size,
        const us nsections) {
    fsTRACE(15);
    dbgassert(filterbank_size <= MAX_SOS_FILTER_BANK_SIZE, 
            "Illegal filterbank size. Max size is " 
            annestr(MAX_SOS_FILTER_BANK_SIZE));

    Sosfilterbank* fb = (Sosfilterbank*) a_malloc(sizeof(Sosfilterbank));

    fb->filterbank_size = filterbank_size;
    dbgassert(nsections < MAX_SOS_FILTER_BANK_NSECTIONS,"Illegal number of sections");
    fb->nsections = nsections;

    /// Allocate filter coefficients matrix
    fb->sos = dmat_alloc(filterbank_size, nsections*6);
    fb->state = dmat_alloc(filterbank_size, nsections*2);
    dmat_set(&(fb->state), 0);

    /// Set all filter coefficients to unit impulse response
    vd imp_response = vd_alloc(6*nsections);
    vd_set(&imp_response,0);
    for(us section = 0;section < nsections; section++) {
        // Set b0 coefficient to 1
        setvecval(&imp_response, 0 + 6*section, 1);
        // Set a0 coefficient to 1
        setvecval(&imp_response, 3 + 6*section, 1);
    }

    // Initialize all filters with a simple impulse response, single pass
    for(us filter_no = 0; filter_no < filterbank_size; filter_no++) {
        Sosfilterbank_setFilter(fb,filter_no,imp_response);
    }
    // Check if coefficients are properly initialized
    // print_dmat(&(fb->sos));
    vd_free(&imp_response);
    feTRACE(15);
    return fb;
}

void Sosfilterbank_setFilter(Sosfilterbank* fb,const us filter_no,
        const vd filter_coefs) {
    fsTRACE(15);
    assertvalidptr(fb);
    assert_vx(&filter_coefs);
    iVARTRACE(15, filter_coefs.n_rows);
    iVARTRACE(15, filter_no);
    dbgassert(filter_no < fb->filterbank_size, "Illegal filter number");
    dbgassert(filter_coefs.n_rows == fb->nsections * 6, 
            "Illegal filter coefficient length");

    dmat *sos = &fb->sos;
    dmat *state = &fb->state;
    us nsections = fb->nsections;

    for(us index=0;index<nsections*6;index++){
        // Copy contents to position in sos matrix
        *getdmatval(sos,filter_no,index) = *getvdval(&filter_coefs,index);
    }

    feTRACE(15);
}

void Sosfilterbank_free(Sosfilterbank* fb) {
    fsTRACE(15);
    assertvalidptr(fb);

    dmat_free(&(fb->sos));
    dmat_free(&(fb->state));

    a_free(fb);
    feTRACE(15);

}

dmat Sosfilterbank_filter(Sosfilterbank* fb,const vd* xs) {

    fsTRACE(15);
    assertvalidptr(fb);
    assert_vx(xs);
    dmat state = fb->state;
    dmat sos = fb->sos;

    us nsections = fb->nsections;
    us filterbank_size = fb->filterbank_size;
    us nsamples = xs->n_rows;

    dmat ys = dmat_alloc(nsamples, filterbank_size);
    /// Copy input signal to output array
    for(us filter=0;filter<filterbank_size;filter++) {
        d_copy(getdmatval(&ys,0,filter),getvdval(xs,0),nsamples,1,1);
    }

    /// Implementation is based on Proakis & Manolakis - Digital Signal
    /// Processing, Fourth Edition, p. 550 
    for(us section=0;section<nsections;section++) {
        /// Obtain state information for current section, and all filters

        for(us filter=0;filter<filterbank_size;filter++) {
            d w1 = *getdmatval(&state,filter,section*2); 
            d w2 = *getdmatval(&state,filter,section*2+1); 

            d b0 = *getdmatval(&sos,filter,section*6+0);
            d b1 = *getdmatval(&sos,filter,section*6+1);
            d b2 = *getdmatval(&sos,filter,section*6+2);
            d a0 = *getdmatval(&sos,filter,section*6+3);
            d a1 = *getdmatval(&sos,filter,section*6+4);
            d a2 = *getdmatval(&sos,filter,section*6+5);

            d* y = getdmatval(&ys, 0, filter);

            for(us sample=0;sample<nsamples;sample++){
                d w0 = *y - a1*w1 - a2*w2;
                d yn = b0*w0 + b1*w1 + b2*w2;
                w2 = w1;
                w1 = w0;
                *y++ = yn;
            }
            *getdmatval(&state,filter,section*2)   = w1;
            *getdmatval(&state,filter,section*2+1) = w2; 

        }
    }



    feTRACE(15);
    return ys;
}
