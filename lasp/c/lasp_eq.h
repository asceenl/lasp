// lasp_eq.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Implementation of an equalizer using the Second Order Sections
// filter bank implementation. 
#pragma once
#ifndef LASP_EQ_H
#define LASP_EQ_H
#include "lasp_sosfilterbank.h"
typedef struct Eq Eq;

/**
 * Initialize an equalizer using the given Filterbank. Note: takes over pointer
 * ownership of fb! Sets levels of all filterbanks initially to 0 dB.
 *
 * @param[in] fb Filterbank to be used in equalizer
 * @return Equalizer handle, NULL on error
 * */
Eq* Eq_create(Sosfilterbank* fb);

/**
 * Equalize a given piece of data using current settings of the levels.
 *
 * @param[in] eq Equalizer handle
 * @param[in] input_data Input data to equalize
 * @return Equalized data. Newly allocated vector. Ownership is transferred.
 * */
vd Eq_equalize(Eq* eq,const vd* input_data);

/**
 * Returns number of channels of the equalizer. Note: takes over pointer
 * ownership of fb!
 *
 * @param[in] eq Equalizer handle
 * */
us Eq_getNLevels(const Eq* eq);

/**
 * Set amplification values for each filter in the equalizer.
 *
 * @param[in] eq Equalizer handle
 * @param[in] levels: Vector with level values for each channel. Should have
 * length equal to the number of filters in the filterbank.
 * */
void Eq_setLevels(Eq* eq, const vd* levels);

/** 
 * Cleans up an existing Equalizer
 *
 * @param[in] eq Equalizer handle
 */
void Eq_free(Eq* eq);

    
        

#endif // LASP_EQ_H
// //////////////////////////////////////////////////////////////////////

