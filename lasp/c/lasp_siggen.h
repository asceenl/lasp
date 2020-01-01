// lasp_siggen.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// Header file for signal generation routines
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_SIGGEN_H
#define LASP_SIGGEN_H
#include "lasp_mat.h"

typedef struct Siggen Siggen;

/**
 * Create a sine wave signal generator
 * 
 * @param[in] fs: Sampling frequency [Hz]
 * @param[in] level: Relative level in [dB], should be between -inf and 0
 * @param[freq] Sine wave frequency [Hz]
 */
Siggen* Siggen_Sinewave_create(const d fs,const d freq,const d level_dB);

/**
 * Create a white noise signal generator
 * 
 * @return Siggen* handle
 */
Siggen* Siggen_Whitenoise_create(const d fs, const d level_dB);

/**
 * Create a pink (1/f) noise signal generator
 * 
 * @param[in] fs: Sampling frequency [Hz]
 * @return Siggen* handle
 */
Siggen* Siggen_Pinknoise_create(const us fs,const d level_dB);

/* Siggen* Siggen_ForwardSweep_create(const d fs,; */
/* Siggen* Siggen_(const d fs,; */

/**
 * Obtain a new piece of signal
 * 
 * @param[in] Siggen* Signal generator private data
 * @param[out] samples Samples to fill. Vector should be pre-allocated
 */
void Siggen_genSignal(Siggen*,vd* samples);
/**
 * Free Siggen data
 * 
 * @param[in] Siggen* Signal generator private data
 */
void Siggen_free(Siggen*);

#endif //LASP_SIGGEN_H
//////////////////////////////////////////////////////////////////////
