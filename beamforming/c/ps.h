// ps.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Single sided power and cross-power spectra computation
// routines.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef PS_H
#define PS_H
#include "window.h"
typedef struct PowerSpectra_s PowerSpectra;


/** 
 * Allocate a PowerSpectra computer
 *
 * @param nfft fft length
 * @param nchannels Number of channels
 * @param wt Windowtype, as defined in window.h
 *
 * @return PowerSpectra handle, NULL on error
 */
PowerSpectra* PowerSpectra_alloc(const us nfft,
                                 const us nchannels,
                                 const WindowType wt);

/** 
 * Compute power spectra.
 *
 * @param ps[in] PowerSpectra handle
 * @param[in] timedata Time data. Should have size nfft*nchannels
 *
 * @param[out] result Here, the result will be stored. Should have
 * size (nfft/2+1)*nchannels^2, such that the Cij at frequency index f
 * can be obtained as result[f,i+j*nchannels]
 *
 */
void PowerSpectra_compute(const PowerSpectra* ps,
                         const dmat* timedata,
                         cmat *result);

/** 
 * Return nfft 
 *
 * @param ps[in] PowerSpectra handle
 *
 * @return nfft
 */
us PowerSpectra_getnfft(const PowerSpectra* ps);

/** 
 * Free PowerSpectra
 *
 * @param[in] ps PowerSpectra handle
 */
void PowerSpectra_free(PowerSpectra* ps);



#endif // PS_H
//////////////////////////////////////////////////////////////////////
