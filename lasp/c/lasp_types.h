// types.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// Typedefs and namespace pollution for stuff that is almost always
// needed.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_TYPES_H
#define LASP_TYPES_H
#include <stddef.h>
// // Branch prediction performance improvement
#if !defined(likely)
#if defined(__GNUC__) && !defined(LASP_DEBUG) 
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#else
#define likely(x)       (x)
#define unlikely(x)     (x)
#endif  // if defined(__GNUC__) && !defined(LASP_DEBUG)

#endif  // !defined(likely)

/// We often use boolean values
#include <stdbool.h>            // true, false
#include <stddef.h>
#include <complex.h>
typedef size_t us;		  /* Size type I always use */

// To change the whole code to 32-bit floating points, change this to
// float.
#if LASP_FLOAT == 32
typedef float d;		/* Shortcut for double */			
typedef float complex c;
#elif LASP_FLOAT == 64
typedef double d;		/* Shortcut for double */			
typedef double complex c;
#else
#error LASP_FLOAT should be either 32 or 64
#endif


/// I need these numbers so often, that they can be in the global
/// namespace.
#define SUCCESS 0
#define INTERRUPTED (-3)
#define MALLOC_FAILED (-1)
#define FAILURE -2


#endif // LASP_TYPES_H
//////////////////////////////////////////////////////////////////////

