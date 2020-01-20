// lasp_slm.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Multi-purpose implementation of a real time Sound Level Meter,
// can be used for full-signal filtering, (fractional) octave band filtering,
// etc.
// //////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_SLM_H
#define LASP_SLM_H
#include "lasp_types.h"
#include "lasp_mat.h"
#include "lasp_sosfilterbank.h"

typedef struct Slm Slm;

#define TAU_FAST (1.0/8.0)
#define TAU_SLOW (1.0)
#define TAU_IMPULSE (35e-3)

/** 
 * Initializes a Sound level meter. NOTE: Sound level meter takes over
 * ownership of pointers to the filterbanks for prefiltering and band-pass
 * filtering! After passing them to the constructor of the Slm, they should not
 * be touched anymore.
 *
 * @param[in] weighting: Sosfiterbank handle, used to pre-filter data. This is
 * in most cases an A-weighting filter, or C-weighting. If NULL, no
 * pre-filtering is done. That can be the case in situations of Z-weighting.
 * @param[in] fb: Sosfiterbank handle, bandpass filters. 
 * @param[in] fs: Sampling frequency in [Hz], used for computing the
 * downsampling factor and size of output arrays.
 * @param[in] tau: Time constant of single pole low pass filter for squared data.
 * Three pre-defined values can be used: TAU_FAST, for fast filtering,
 * TAU_SLOW, for slow filtering, TAU_IMPULSE for impulse filtering
 * @param[in] ref_level: Reference level when computing dB's. I.e. P_REF_AIR for
 * sound pressure levels in air
 * @param[out] downsampling_fac: Here, the used downsampling factor is stored
 * which is used on returning data. If the value is for example 10, the
 * 'sampling' frequency of output data from `Slm_run` is 4800 is fs is set to
 * 48000. This downsampling factor is a function of the used time weighting.
 *
 * @return Slm: Handle of the sound level meter, NULL on error.
 */
Slm* Slm_create(Sosfilterbank* weighting,Sosfilterbank* bandpass,
                const d fs,
                const d tau,
                const d ref_level,
                us* downsampling_fac);

/**
 * Run the sound level meter on a piece of time data.
 *
 * @param[in] slm: Slm handle
 * @param[in] input_data: Vector of input data samples.
 *
 * @return Output result of Sound Level values in [dB], for each bank in the filter
 * bank.
 */
dmat Slm_run(Slm* slm,
             vd* input_data);
/** 
 * Cleans up an existing Slm
 *
 * @param f slm handle
 */
void Slm_free(Slm* f);


#endif // LASP_SLM_H
//////////////////////////////////////////////////////////////////////
