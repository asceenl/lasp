// fftpack.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// Header file for FFT routines of fftpack
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef FFTPACK_H
#define FFTPACK_H
#include "tracer.h"
#include "ascee_alloc.h"
#include "npy_fftpack.h"

/*
 * Fortran has two kind of routines: functions and subroutines. A
 Fortran function is a C function that returns a single value.

 A * subroutine called from C is in fact a C function
 which returns void.
 */
#if ASCEE_DOUBLE_PRECISION
// These are all subroutines
extern void dffti_ (int *nfft,d* wsave);
extern void dfftf_ (int *nfft,d* r,d* wsave);
#else
extern void rffti_ (int *nfft,d* wsave);
extern void rfftf_ (int *nfft,d* r,d* wsave);
#endif // ASCEE_DOUBLE_PRECISION

typedef struct Fftr_s {
    d *work_area;
    d* wsave_ptr;
    int nfft;
} Fftr;

/** 
 * 
 *
 * @param nfft the length of the sequence to be transformed

 * @param wsave a work array which must be dimensioned at least
 * 2*nfft+15.  the same work array can be used for both rfftf and
 * rfftb as long as n remains unchanged. different wsave arrays
 * are required for different values of n. the contents of wsave
 * must not be changed between calls of rfftf or rfftb.
 */

Fftr* Fftr_alloc(int nfft) {
    fsTRACE(15);

    Fftr* fftr = a_malloc(sizeof(fftr));
    dbgassert(nfft>0,"Invalid nfft")
    dbgassert(fftr,ALLOCFAILED "Fftr_alloc");
    fftr->work_area = a_malloc(sizeof(d)*(3*nfft+15));
    fftr->wsave_ptr = &fftr->work_area[nfft];
    fftr->nfft = nfft;
    #if ASCEE_DOUBLE_PRECISION
    // dffti_(&nfft,fftr->wsave);
    npy_rffti(nfft,fftr->wsave_ptr);
    #else
    // rffti_(&nfft,fftr->wsave);
    #endif

    feTRACE(15);
    return fftr;
}
void Fftr_free(Fftr* fftr) {
    dbgassert(fftr,NULLPTRDEREF "Fftr_free");
    a_free(fftr->work_area);
    a_free(fftr);
}


/** 
 * 
 *
 * @param nfft 
 * @param wsave 
 * @param input 
 * @param work
 * @param result 
 */
static inline void Fftr_fftr(Fftr* fftr,d* input,c* result) {


    // Copy contents of input to the work area
    d_copy(fftr->work_area,input,fftr->nfft);

    int nfft = fftr->nfft;
    
    #if ASCEE_DOUBLE_PRECISION
    // dfftf_(&nfft,fftr->work,fftr->wsave);
    npy_rfftf(nfft,fftr->work_area,fftr->wsave_ptr);
    #else
    NOT TESTED
    rfftf_(&nfft,fftr->work_area,fftr->wsave_ptr);
    #endif

    result[0] = fftr->work_area[0];
    d_copy((d*) (&result[1]),&fftr->work_area[1],nfft);
    // Not portable way of setting imaginary part to zero. Works with
    // gcc, though.
    __imag__ result[nfft/2] = 0;
    
}



#endif // FFTPACK_H
//////////////////////////////////////////////////////////////////////
