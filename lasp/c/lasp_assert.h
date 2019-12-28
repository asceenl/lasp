// ascee_assert.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// Basic tools for debugging using assert statements including text.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_ASSERT_H
#define LASP_ASSERT_H

#define OUTOFBOUNDSMATR "Out of bounds access on matrix row"
#define OUTOFBOUNDSMATC "Out of bounds access on matrix column"
#define OUTOFBOUNDSVEC "Out of bounds access on vector"
#define SIZEINEQUAL "Array sizes not equal"
#define ALLOCFAILED "Memory allocation failure in: "
#define NULLPTRDEREF "Null pointer dereference in: "

#ifdef LASP_DEBUG
#include "lasp_types.h"


void DBG_AssertFailedExtImplementation(const char* file,
                                       const us line,
                                       const char* string);

#define dbgassert(assertion, assert_string) \
	if (!(assertion)) \
    { \
    	DBG_AssertFailedExtImplementation(__FILE__, __LINE__, assert_string );  \
    }

#define assertvalidptr(ptr) dbgassert(ptr,NULLPTRDEREF)

#else // LASP_DEBUG not defined

#define dbgassert(assertion, assert_string)

#define assertvalidptr(ptr) dbgassert(ptr,NULLPTRDEREF)

#endif  // LASP_DEBUG

#endif // LASP_ASSERT_H
//////////////////////////////////////////////////////////////////////
