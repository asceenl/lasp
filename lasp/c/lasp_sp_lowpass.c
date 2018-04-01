// lasp_sp_lowpass.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// Single pole lowpass IIR filter implementation.
//////////////////////////////////////////////////////////////////////
#include "lasp_sp_lowpass.h"
#include "lasp_alloc.h"

typedef struct SPLowpass_s {

    d a;
    d b;

    d xlast,ylast;

} SPLowpass;

SPLowpass* SPLowpass_create(d fs,d tau) {
    fsTRACE(15);

    d tau_i = 1/tau;
    d T = 1/fs;

    if(fs <= 0) {
        WARN("Invalid sampling frequency");
        return NULL;
    }
    else if(T >= tau ) {
        WARN("Invalid tau, should be (much) larger than sampling"
             " time (1/fs)");
        return NULL;
    }


    SPLowpass* lp = a_malloc(sizeof(SPLowpass));
    lp->xlast = 0;
    lp->ylast = 0;

    lp->a = tau_i/(2*fs+tau_i);
    lp->b = (2*fs-tau_i)/(2*fs+tau_i);

    feTRACE(15);
    return lp;
}

vd SPLowpass_filter(SPLowpass* lp,
                    const vd* input) {

    fsTRACE(15);
    dbgassert(lp && input,NULLPTRDEREF);
    assert_vx(input);
    us input_size = input->n_rows;

    if(input_size == 0) {
        return vd_alloc(0);
    }
    
    vd output = vd_alloc(input_size);
    const d xlast = lp->xlast;
    const d ylast = lp->ylast;
    const d a = lp->a;
    const d b = lp->b;
    
    *getvdval(&output,0) = a*(xlast + *getvdval(input,0)) +
        b*ylast;
    
    for(us i=1;i<input_size;i++) {

        *getvdval(&output,i) = a*(*getvdval(input,i-1) +
                                  *getvdval(input,i))  +
            b*(*getvdval(&output,i-1));

    }

    lp->xlast = *getvdval(input  ,input_size-1);
    lp->ylast = *getvdval(&output,input_size-1);

    feTRACE(15);
    return output;
}

void SPLowpass_free(SPLowpass* lp) {
    fsTRACE(15);
    dbgassert(lp,NULLPTRDEREF);
    a_free(lp);
    feTRACE(15);
}

//////////////////////////////////////////////////////////////////////
