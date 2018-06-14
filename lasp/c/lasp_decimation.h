// lasp_decimation.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Sample decimator, works on a whole block of
// (uninterleaved) time samples at once. Decimates (downsamples and
// low-pass filters by a factor given integer factor.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_DECIMATION_H
#define LASP_DECIMATION_H
#include "lasp_types.h"
#include "lasp_mat.h"

typedef struct Decimator_s Decimator;

typedef enum DEC_FAC_e {
  DEC_FAC_4 = 0, // Decimate by a factor of 4
} DEC_FAC;

/**
 * Create a decimation filter for a given number of channels and
 * decimation factor
 *
 * @param nchannels Number of channels

 * @param d Decimation factor. Should be one of the implemented
 * ones. (TODO: add more. At this point we have only a decimation
 * factor of 4 implemented)
 *
 * @return Decimator handle. NULL on error
 */
Decimator* Decimator_create(us nchannels,DEC_FAC d);

/** 
 * Decimate given samples
 *
 * @param samples 
 *
 * @return Decimated samples. Can be an empty array.
 */
dmat Decimator_decimate(Decimator* dec,const dmat* samples);


d Decimator_get_cutoff(Decimator*);
/** 
 * Free memory corresponding to Decimator
 *
 * @param dec Decimator handle.
 */
void Decimator_free(Decimator* dec);

#endif // LASP_DECIMATION_H
//////////////////////////////////////////////////////////////////////
