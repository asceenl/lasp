// lasp_sosfilterbank.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Implemententation of a discrete filterbank using cascaded
// second order sections (sos), also called BiQuads.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_FILTERBANK_H
#define LASP_FILTERBANK_H
#include "lasp_types.h"
#include "lasp_mat.h"

#define MAX_SOS_FILTER_BANK_SIZE 40
#define MAX_SOS_FILTER_BANK_NSECTIONS 6

typedef struct Sosfilterbank Sosfilterbank;

/** 
 * Initializes a Sosfilterbank. Sets all coefficients in such a way that the
 * filter effectively does nothing (unit impulse response).
 */
Sosfilterbank* Sosfilterbank_create(const us filterbank_size,
                                    const us nsections);

/**
 * Returns the number of channels in the filterbank (the filberbank size).
 *
 * @param[in] fb: Filterbank handle
 * @return The number of filters in the bank
 * */
us Sosfilterbank_getFilterbankSize(const Sosfilterbank* fb);

/**
 * Initialize the filter coeficients in the filterbank
 *
 * @param fb: Filterbank handle
 * @param filter_no: Filter number in the bank
 * @param coefss: Array of filter coefficients. Should have a length of 
 * nsections x 6, for each of the sections, it contains (b0, b1, b2, a0, 
 * a1, a2), where a are the numerator coefficients and b are the denominator 
 * coefficients. 
 *
*/
void Sosfilterbank_setFilter(Sosfilterbank* fb,const us filter_no,
                             const vd coefs);

/** 
 * Filters x using h, returns y
 *
 * @param x Input time sequence block. Should have at least one sample.

 * @return Filtered output in an allocated array. The number of
 * columns in this array equals the number of filters in the
 * filterbank. The number of output samples is equal to the number of
 * input samples in x.
 */
dmat Sosfilterbank_filter(Sosfilterbank* fb,
                          const vd* x);

/** 
 * Cleans up an existing filter bank.
 *
 * @param f Filterbank handle
 */
void Sosfilterbank_free(Sosfilterbank* f);


#endif // LASP_FILTERBANK_H
//////////////////////////////////////////////////////////////////////
