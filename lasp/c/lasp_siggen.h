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
#include "lasp_sosfilterbank.h"

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
 * Create a Noise signal generator. If no Sosfilterbank is provided, it will
 * create white noise. Otherwise, the noise is 'colored' using the filterbank
 * given in the constructor. Note that the pointer to this filterbank is
 * *STOLEN*!.
 *
 * @param[in] fs: Sampling frequency [Hz]
 * @param[in] level_dB: Relative level [dB]
 * @param[in]
 * 
 * @return Siggen* handle
 */
Siggen* Siggen_Noise_create(const d fs, const d level_dB, Sosfilterbank* colorfilter);

// Define this flag to repeat a forward sweep only, or backward only. If not
// set, we do a continuous sweep
#define SWEEP_FLAG_FORWARD      1
#define SWEEP_FLAG_BACKWARD     2

// Types of sweeps
#define SWEEP_FLAG_LINEAR       4
#define SWEEP_FLAG_EXPONENTIAL  8
#define SWEEP_FLAG_HYPERBOLIC  16

/**
 * Create a forward sweep
 * 
 * @param[in] fs: Sampling frequency [Hz]
 * @param[in] fl: Lower frequency [Hz]
 * @param[in] fl: Upper frequency [Hz]
 * @param[in] Ts: Sweep period [s]
 * @param[in] sweep_flags: Sweep period [s]
 * @param[in] level: Relative level in [dB], should be between -inf and 0
 * @return Siggen* handle
 */
Siggen* Siggen_Sweep_create(const d fs,const d fl,const d fu,
                            const d Ts, const us sweep_flags,
                            const d level);

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
