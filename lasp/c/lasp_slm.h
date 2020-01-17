// lasp_slm.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Implementation of a real time Sound Level Meter, based on
// given slm.
// //////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_slm_H
#define LASP_slm_H
#include "lasp_types.h"
#include "lasp_mat.h"
#include "lasp_sosfilterbank.h"

typedef struct Slm Slm;

/** 
 * Initializes a Sound level meter.
 *
 * @param fb: Sosfiterbank handle, used to pre-filter data
 * @param T:  Single pole low pass filter time constant to use.
 */
slm* slm_create(Sosfilterbank* fb);

/**
 * Initialize the filter coeficients in the slm
 *
 * @param fb: slm handle
 * @param filter_no: Filter number in the bank
 * @param coefss: Array of filter coefficients. Should have a length of 
 * nsections x 6, for each of the sections, it contains (b0, b1, b2, a0, 
 * a1, a2), where a are the numerator coefficients and b are the denominator 
 * coefficients. 
 *
*/
void slm_setFilter(slm* fb,const us filter_no,
                             const vd coefs);

/** 
 * Filters x using h, returns y
 *
 * @param x Input time sequence block. Should have at least one sample.

 * @return Filtered output in an allocated array. The number of
 * columns in this array equals the number of filters in the
 * slm. The number of output samples is equal to the number of
 * input samples in x.
 */
dmat slm_filter(slm* fb,
                          const vd* x);

/** 
 * Cleans up an existing filter bank.
 *
 * @param f slm handle
 */
void slm_free(slm* f);


#endif // LASP_slm_H
//////////////////////////////////////////////////////////////////////
