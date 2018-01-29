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

struct PowerSpectra_s;
struct AvPowerSpectra_s;

typedef struct PowerSpectra_s PowerSpectra;
typedef struct AvPowerSpectra_s AvPowerSpectra;


PowerSpectra* PowerSpectra_alloc(const us nfft,
                                 const us nchannels,
                                 const WindowType wt);

/** 
 * Compute power spectra, returns to a complex array
 *
 * @param[in] timedata Time data, should be of size nfft*nchannels
 * @param [out] Cross-power spectra, array should be of size
 * (nfft/2+1) x (nchannels*nchannels), such that the cross spectra of
 * channel i with channel j at can be found as
 * getvcval(0,i+j*nchannels).
 * @return status code, SUCCESS on succes.
 */
int PowerSpectra_compute(const PowerSpectra*,
                         const dmat* timedata,
                         cmat *result);

/** 
 * Free storage of the PowerSpectra
 */
void PowerSpectra_free(PowerSpectra*);


// AvPowerSpectra* AvPowerSpectra_alloc(const us nfft,
//                                      const us nchannels,
//                                      const d overlap_percentage,
//                                      const WindowType wt);
                                 

/** 
 * Return the current number of averages taken to obtain the result.
 *
 * @return The number of averages taken.
 */
us AvPowerSpectra_getAverages(const AvPowerSpectra*);


int AvPowerSpectra_addTimeData(AvPowerSpectra*,
                             const dmat* timedata);

/** 
 * Free storage of the AvPowerSpectra
 */
void AvPowerSpectra_free(AvPowerSpectra*);

#endif // PS_H
//////////////////////////////////////////////////////////////////////
