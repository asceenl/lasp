// fft.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// Interface to the FFT library, multiple channel FFT's
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef FFT_H
#define FFT_H
#include "types.h"
#include "ascee_math.h"

/**
 * Perform forward FFT's on real time data.
 * 
 */
typedef struct Fft_s Fft;

/** 
 * Construct an Fft object
 *
 * @param nfft Nfft size
 * @param nchannels Number of channels
 *
 * @return Pointer to Fft handle, NULL on error
 */
Fft* Fft_alloc(const us nfft,const us nchannels);

/** 
 * Returns the number of channels for this Fft instance
 *
 * @return nchannels
 */
us Fft_nchannels(const Fft*);

/** 
 * Returns the nfft for this Fft instance
 *
 * @return nfft
 */
us Fft_nfft(const Fft*);


/** 
 * Compute the fft of the data matrix, first axis is assumed to be
 * the time axis.
 *
 * @param[in] timedata Input time data pointer, such that
 * data[i*nfft+j) is the i-th channel from data stream j.
 *
 * @param[out] result: Fft't data, size should be (nfft/2+1)*nchannels
 */
void Fft_fft(const Fft*,const dmat* timedata,cmat* result);

void Fft_free(Fft* fft);

#endif // FFT_H
//////////////////////////////////////////////////////////////////////
