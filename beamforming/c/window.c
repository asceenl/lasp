// window.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////

#include "window.h"
#include "signals.h"

/** 
 * Compute the Hann window
 *
 * @param i index
 * @param N Number of indices
 */
static d hann(us i,us N) {
    dbgassert(i<N,"Invalid index for window function Hann");
    return d_pow(d_sin(number_pi*i/(N-1)),2);
}
/** 
 * Compute the Hamming window
 *
 * @param i index
 * @param N Number of indices
 */
static d hamming(us i,us N) {
    dbgassert(i<N,"Invalid index for window function Hamming");
    d alpha = 25.0/46.0;
    return alpha-(1-alpha)*d_cos(2*number_pi*i/(N-1));
}
/** 
 * Compute the Blackman window
 *
 * @param i index
 * @param N Number of indices
 */
static d blackman(us i,us N) {
    dbgassert(i<N,"Invalid index for window function Blackman");
    d a0 = 7938./18608.;
    d a1 = 9240./18608.;
    d a2 = 1430./18608.;
    return a0-a1*d_cos(2*number_pi*i/(N-1))+a2*d_cos(4*number_pi*i/(N-1));
}
/** 
 * Compute the Rectangular window
 *
 * @param i index
 * @param N Number of indices
 */
static d rectangle(us i,us N) {
    dbgassert(i<N,"Invalid index for window function Hann");
    return 1.0;
}

int window_create(const WindowType wintype,vd* result,d* win_pow) {

    us nfft = result->size;
    d (*win_fun)(us,us);
    switch (wintype) {
    case Hann: {
        win_fun = hann;
        break;
    }
    case Hamming: {
        win_fun = hamming;
        break;
    }
    case Blackman: {
        win_fun = blackman;
        break;
    }
    case Rectangular: {
        win_fun = rectangle;
        break;
    }
    default:
        WARN("Unkown window function");
        return FAILURE;
        break;
    }
    us index;
    for(index=0;index<nfft;index++) {
        /* Compute the window function value */
        d val = win_fun(index,nfft);

        /* Set the value in the vector */
        setvecval(result,index,val);
    }

    *win_pow = signal_power(result);

    /* Store window power in result */
    
    return SUCCESS;
}


//////////////////////////////////////////////////////////////////////
