// types.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// Typedefs and namespace pollution for stuff that is almost always
// needed.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef TYPES_H
#define TYPES_H

/// We often use boolean values
#include <stdbool.h>            // true, false
#include <stddef.h>
#include <complex.h>
typedef size_t us;		  /* Size type I always use */

// To change the whole code to 32-bit floating points, change this to
// float.
#if ASCEE_FLOAT == 32
typedef float d;		/* Shortcut for double */			
typedef float complex c;
#elif ASCEE_FLOAT == 64
typedef double d;		/* Shortcut for double */			
typedef double complex c;
#else
#error ASCEE_FLOAT should be either 32 or 64
#endif


/// I need these numbers so often, that they can be in the global
/// namespace.
#define SUCCESS 0
#define INTERRUPTED (-3)
#define MALLOC_FAILED (-1)
#define FAILURE -2



#endif // ASCEE_TYPES_H
//////////////////////////////////////////////////////////////////////

