// window.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_WINDOW_H
#define LASP_WINDOW_H
#include "lasp_math.h"

typedef enum {
    Hann = 0,
    Hamming = 1,
    Rectangular = 2,    
    Bartlett = 3,
    Blackman = 4,

} WindowType;

/** 
 * Create a Window function, store it in the result
 *
 * @param[in] wintype Enumerated type, corresponding to the window
 * function. 
 * @param[out] result Vector where the window values are stored
 * @param[out] win_power Here, the overall power of the window will be
 * returned.
 * 
 * @return status code, SUCCESS on success.
 */
int window_create(const WindowType wintype,vd* result,d* win_power);


#endif // LASP_WINDOW_H
//////////////////////////////////////////////////////////////////////
