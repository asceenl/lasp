// lasp_firfilterbank.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Implemententation of a discrete FIR filterbank using fast
// convolution and the overlap-save (overlap-scrap method). Multiple
// filters can be applied to the same input data (*filterbank*).
// Implementation is computationally efficient, as the forward FFT is
// performed only over the input data, and the backwards transfer for
// each filter in the filterbank.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_FIRFILTERBANK_H
#define LASP_FIRFILTERBANK_H
#include "lasp_types.h"
#include "lasp_mat.h"
typedef struct Firfilterbank_s Firfilterbank;

/** 
 * Initializes a fast convolution filter bank and returns a Firfilterbank
 * handle. The nfft will be chosen to be at least four times the
 * length of the FIR filters.
 *
 * @param h: matrix with filter coefficients of each of the
 * filters. First axis is the axis of the filter coefficients, second
 * axis is the filter number. Maximum length of the filter is nfft/2.
 * 
 * @param nfft: FTT length for fast convolution. For good performance,
 * nfft should be chosen as the nearest power of 2, approximately four
 * times the filter lengths. For the lowest possible latency, it is
 * better to set nfft at twice the filter length.
 *
 * @return Firfilterbank handle, NULL on error.
 */
Firfilterbank* Firfilterbank_create(const dmat* h,const us nfft);

/** 
 * Filters x using h, returns y
 *
 * @param x Input time sequence block. Should have at least one sample.

 * @return Filtered output in an allocated array. The number of
 * columns in this array equals the number of filters in the
 * filterbank. The number of output samples is equal to the number of
 * input samples in x.
 */
dmat Firfilterbank_filter(Firfilterbank* fb,
                       const vd* x);

/** 
 * Cleans up an existing filter bank.
 *
 * @param f Filter handle
 */
void Firfilterbank_free(Firfilterbank* f);


#endif // LASP_FIRFILTERBANK_H
//////////////////////////////////////////////////////////////////////
