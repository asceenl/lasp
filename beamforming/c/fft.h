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
 *
 * @return Pointer to Fft handle, NULL on error
 */
Fft* Fft_alloc(const us nfft);
/** 
 * Returns the nfft for this Fft instance
 *
 * @return nfft
 */
us Fft_nfft(const Fft*);

/** 
 * Compute the fft of a single channel.
 *
 * @param[in] fft Fft handle.
 *
 * @param[in] timedata Input time data pointer, should have size nfft
 * @param[out] result Pointer to result vector should have size
 * nfft/2+1
 */
void Fft_fft_single(const Fft* fft,const vd* timedata,vc* result);

/** 
 * Compute the fft of the data matrix, first axis is assumed to be
 * the time axis.
 *
 * @param[in] fft Fft handle.

 * @param[in] timedata Input time data. First axis is assumed to be
 * the time, second the channel number.
 *
 * @param[out] result: Fft't data, should have size (nfft/2+1) *
 * nchannels
 */
void Fft_fft(const Fft* fft,const dmat* timedata,cmat* result);

/** 
 * 
 *
 * @param fft 
 */
void Fft_free(Fft* fft);

#endif // FFT_H
//////////////////////////////////////////////////////////////////////
