// lasp_sp_lowpass.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Single-pole low pass IIR filter. Used for example for
// fast and fast and slow averaging of the square of the squared
// A-weighted acoustic pressure. This class uses the bilinear
// transform to derive the filter coefficients.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_SP_LOWPASS_H
#define LASP_SP_LOWPASS_H
#include "lasp_types.h"
#include "lasp_mat.h"

typedef struct SPLowpass_s SPLowpass;

/** 
 * Create a single pole lowpass IIR filter.
 *
 * @param fs Sampling frequency [Hz]

 * @param tau Time constant of the lowpass filter. Should be >0,
 * otherwise an error occurs
 *
 * @return Pointer to dynamically allocated lowpass filter. NULL on
 * error.
 */
SPLowpass* SPLowpass_create(d fs,d tau);

/** 
 * Use the filter to filter input data
 *
 * @param lp Lowpass filter handlen
 * @param input Vector of input samples.
 *
 * @return Output data. Length is equal to input at all cases.
 */
vd SPLowpass_filter(SPLowpass* lp,
                    const vd* input);

/** 
 * Free a single pole lowpass filter
 *
 * @param lp 
 */
void SPLowpass_free(SPLowpass* lp);


#endif // LASP_SP_LOWPASS_H
//////////////////////////////////////////////////////////////////////
