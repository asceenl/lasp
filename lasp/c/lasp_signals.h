// lasp_signals.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// Several signal functions
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_SIGNALS_H
#define LASP_SIGNALS_H
#include "lasp_math.h"

/** 
 * Compute the signal power, that is \f$ \frac{1}{N} \sum_{i=0}^{N-1}
 * v_i^2 \f$
 *
 * @param[in] signal Signal to compute the power of.
 * @return the signal power
 */
static inline d signal_power(vd* signal) {
    d res = 0;
    for(us i=0;i<signal->size;i++) {
        res+= d_pow(*getvdval(signal,i),2);
    }
    res /= signal->size;
    return res;
}



#endif // LASP_SIGNALS_H
//////////////////////////////////////////////////////////////////////
