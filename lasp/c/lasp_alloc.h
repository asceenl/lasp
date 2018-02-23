// lasp_alloc.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// memory allocation functions.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_ALLOC_H
#define LASP_ALLOC_H
#include <malloc.h>
#include "lasp_tracer.h"
/**
 * Reserved words for memory allocation. Can be changed to something
 * else when required. For example for debugging purposes.
 */
static inline void* a_malloc(size_t nbytes) {
    void* ptr = malloc(nbytes);
    if(!ptr) {
        FATAL("Memory allocation failed. Exiting");
    }
    return ptr;
}
#define a_free free
#define a_realloc realloc

#endif // LASP_ALLOC_H
//////////////////////////////////////////////////////////////////////
