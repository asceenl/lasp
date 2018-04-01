// aps.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_APS_H
#define LASP_APS_H
#include "lasp_types.h"
#include "lasp_mat.h"
#include "lasp_window.h"

typedef enum {
    Linear=0,
    Exponential=1
} TimeWeighting;


typedef struct AvPowerSpectra_s AvPowerSpectra;

/** 
 * Allocates a AvPowerSpectra object which is used to compute an
 * averaged power spectra from given input time samples.
 *
 * @param[in] nfft Size of the fft,
 *
 * @param[in] nchannels Number of channels of time data to allocate
 * for.
 *
 * @param[in] overlap_percentage. Should be 0<= overlap_pct < 100. The
 * overlap percentage will be coerced, as the offset will be an
 * integer number of samples at all cases. Therefore, the real overlap
 * percentage can be obtained later on
 *
 * @param wt 
 *
 * @return 
 */
AvPowerSpectra* AvPowerSpectra_alloc(const us nfft,
                                     const us nchannels,
                                     const d overlap_percentage,
                                     const WindowType wt,
                                     const vd* spectrogram_weighting);


/** 
 * Computes the real overlap percentage, from the integer overlap
 * offset.
 *
 * @param aps 
 */
d AvPowerSpectra_realoverlap_pct(const AvPowerSpectra* aps);

/** 
 * Return the current number of averages taken to obtain the result.
 *
 * @return The number of averages taken.
 */
us AvPowerSpectra_getAverages(const AvPowerSpectra*);

/** 
 * Add time data to this Averaged Power Spectra. Remembers the part
 * that should be overlapped with the next time data.
 *
 * @param aps AvPowerSpectra handle 
 *
 * @param timedata Pointer to timedata buffer. Number of rows SHOULD
 * *at least* be nfft. Number of columns should exactly be nchannels.
 *
 * @return pointer to the averaged power spectra buffer that is being
 * written to. Pointer is valid until AvPowerSpectra_free() is called.
 */
cmat* AvPowerSpectra_addTimeData(AvPowerSpectra* aps,
                                 const dmat* timedata);

/** 
 * Free storage of the AvPowerSpectra
 */
void AvPowerSpectra_free(AvPowerSpectra*);


#endif // LASP_APS_H
//////////////////////////////////////////////////////////////////////
