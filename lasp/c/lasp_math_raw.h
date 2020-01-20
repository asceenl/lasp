// lasp_math_raw.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Raw math routines working on raw arrays of floats and
// complex numbers.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_MATH_RAW_H
#define LASP_MATH_RAW_H
#include "lasp_assert.h"
#include "lasp_tracer.h"
#include "lasp_types.h"
#include <math.h>

#if LASP_USE_BLAS == 1
#include <cblas.h>
#elif LASP_USE_BLAS == 0
#else
#error "LASP_USE_BLAS should be set to either 0 or 1"
#endif

#ifdef LASP_DOUBLE_PRECISION
#define c_real creal
#define c_imag cimag
#define d_abs fabs
#define c_abs cabs
#define c_conj conj
#define d_atan2 atan2
#define d_acos acos
#define d_sqrt sqrt
#define c_exp cexp
#define d_sin sin
#define d_cos cos
#define d_pow pow
#define d_log10 log10

#else  // LASP_DOUBLE_PRECISION not defined
#define c_conj conjf
#define c_real crealf
#define c_imag cimagf
#define d_abs fabsf
#define c_abs cabsf
#define d_atan2 atan2f
#define d_acos acosf
#define d_sqrt sqrtf
#define c_exp cexpf
#define d_sin sinf
#define d_cos cosf
#define d_pow powf
#define d_log10 log10f
#endif // LASP_DOUBLE_PRECISION

#ifdef M_PI
static const d number_pi = M_PI;
#else
static const d number_pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;
#endif

/** 
 * Set all elements in an array equal to val
 *
 * @param to 
 * @param val 
 * @param size 
 */
static inline void d_set(d to[],d val,us size) {
    for(us i=0;i<size;i++) {
        to[i]=val;
    }
}
/** 
 * Set all elements in an array equal to val
 *
 * @param to 
 * @param val 
 * @param size 
 */
static inline void c_set(c to[],c val,us size) {
    for(us i=0;i<size;i++) {
        to[i]=val;
    }
}

/** 
 * Return the maximum of two doubles
 *
 * @param a value 1
 * @param b value 2
 * 
 * @returns the maximum of value 1 and 2
 */
static inline d d_max(const d a,const d b) {
    return a>b?a:b;
}


/** 
 * Return the dot product of two arrays, one of them complex-valued,
 * the other real-valued
 *
 * @param a the complex-valued array
 * @param b the real-valued array
 * @param size the size of the arrays. *Should be equal-sized!*
 *
 * @return the dot product
 */
static inline c cd_dot(const c a[],const d b[],us size){
    c result = 0;
    us i;
    for(i=0;i<size;i++){
        result+=a[i]*b[i];
    }
    return result;
}
/** 
 * Return the dot product of two complex-valued arrays. Wraps BLAS
 * when LASP_USE_BLAS == 1.
 *
 * @param a complex-valued array
 * @param b complex-valued array
 * @param size the size of the arrays. *Should be equal-sized!*
 *
 * @return the dot product
 */
static inline c cc_dot(const c a[],const c b[],us size){
    #if LASP_USE_BLAS == 1
    WARN("CBlas zdotu not yet tested");
    #if LASP_DOUBLE_PRECISION
    // assert(0);
    return cblas_zdotu(size,(d*) a,1,(d*) b,1);
    #else
    return cblas_cdotu(size,(d*) a,1,(d*) b,1);
    #endif
    #else
    c result = 0;
    us i;
    for(i=0;i<size;i++){
        result+=a[i]*b[i];
    }
    return result;
    #endif
}

/** 
 * Compute the dot product of two real arrays.
 *
 * @param a First array.
 * @param b Second array.
 * @param size Size of the arrays.
 * @return The result.
 */
static inline d d_dot(const d a[],const d b[],const us size){
    #if LASP_USE_BLAS == 1
    #if LASP_DOUBLE_PRECISION
    return cblas_ddot(size,a,1,b,1);
    #else  // Single precision function
    return cblas_sdot(size,a,1,b,1);
    #endif
    #else  // No BLAS, do it manually
    d result = 0;
    us i;
    for(i=0;i<size;i++){
        result+=a[i]*b[i];
    }
    return result;
    #endif
}


/** 
 * Copy array of floats.
 *
 * @param to : Array to write to
 * @param from : Array to read from
 * @param size : Size of arrays
 * @param to_inc : Mostly equal to 1, the stride of the array to copy to
 * @param from_inc : Mostly equal to 1, the stride of the array to copy from
 */
static inline void d_copy(d to[],
                          const d from[],
                          const us size,
                          const us to_inc,
                          const us from_inc){
    #if LASP_USE_BLAS == 1
    cblas_dcopy(size,from,from_inc,to,to_inc);
    #else
    us i;
    for(i=0;i<size;i++)
        to[i*to_inc] = from[i*from_inc];
    #endif
}


/** 
 * Copy array of floats to array of complex floats. Imaginary part set
 * to zero.
 *
 * @param to : Array to write to
 * @param from : Array to read from
 * @param size : Size of arrays
 */
static inline void cd_copy(c to[],const d from[],const us size) {
    us i;
    for(i=0;i<size;i++) {
        to[i] = (c) (from[i]);
        dbgassert(cimag(to[i]) == 0,"Imaginary part not equal to zero");
    }
}

/** 
 * Copy float vector to complex vector. Imaginary part set
 * to zero.
 *
 * @param to : Vector to write to
 * @param from : Vector to read from
 */
static inline void c_copy(c to[],const c from[],const us size){
    
    #if LASP_USE_BLAS == 1
    #if LASP_DOUBLE_PRECISION
    cblas_zcopy(size,(d*) from,1,(d*) to,1);
    #else
    cblas_ccopy(size,(d*) from,1,(d*) to,1);
    #endif
    #else
    us i;
    for(i=0;i<size;i++)
        to[i] = from[i];
    #endif
}
/** 
 * Multiply y with fac, and add result to x
 *
 * @param x[in,out] Array to add to
 * @param y[in] Array to add to x
 * @param[in] fac Factor with which to multiply y
 * @param[in] size Size of the arrays
 */
static inline void d_add_to(d x[],const d y[],
                            const d fac,const us size){
    #if LASP_USE_BLAS == 1
    #if LASP_DOUBLE_PRECISION
    cblas_daxpy(size,fac,y,1,x,1);
    #else
    cblas_saxpy(size,fac,y,1,x,1);
    #endif
    #else
    us i;
    for(i=0;i<size;i++)
        x[i]+=fac*y[i];
    #endif
}
/** 
 * x = x + fac*y
 *
 * @param[in,out] x Array to add to
 * @param[in] y Array to add to x
 * @param[in] fac Factor with which to multiply y
 * @param[in] size Size of the arrays
 */
static inline void c_add_to(c x[],const c y[],
                            const c fac,const us size){
    fsTRACE(15);
    #if LASP_USE_BLAS == 1
    #if LASP_DOUBLE_PRECISION
    cblas_zaxpy(size,(d*) &fac,(d*) y,1,(d*) x,1);
    #else
    cblas_caxpy(size,(d*) &fac,(d*) y,1,(d*) x,1);
    #endif
    #else
    us i;
    for(i=0;i<size;i++)
        x[i]+=fac*y[i];
    #endif
    feTRACE(15);
}

/** 
 * Scale an array of doubles
 *
 * @param a array
 * @param scale_fac scale factor
 * @param size size of the array
 */
static inline void d_scale(d a[],const d scale_fac,us size){
    #if LASP_USE_BLAS == 1
    #if LASP_DOUBLE_PRECISION    
    cblas_dscal(size,scale_fac,a,1);
    #else
    cblas_sscal(size,scale_fac,a,1);
    #endif
    #else
    us i;
    for(i=0;i<size;i++)
        a[i]*=scale_fac;
    #endif
}


/** 
 * Scale an array of complex floats
 *
 * @param a array
 * @param scale_fac scale factor
 * @param size size of the array
 */
static inline void c_scale(c a[],const c scale_fac,us size){
    #if LASP_USE_BLAS == 1
    // Complex argument should be given in as array of two double
    // values. The first the real part, the second the imaginary
    // part. Fortunately the (c) type stores the two values in this
    // order. To be portable and absolutely sure anything goes well,
    // we convert it explicitly here.

    #if LASP_DOUBLE_PRECISION
    cblas_zscal(size,(d*) &scale_fac,(d*) a,1);
    #else
    cblas_cscal(size,(d*) &scale_fac,(d*) a,1);
    #endif
    #else
    us i;
    for(i=0;i<size;i++)
        a[i]*=scale_fac;
    #endif
}


/** 
 * Compute the maximum value of an array
 *
 * @param a array
 * @param size size of the array
 * @return maximum
 */
static inline d darray_max(const d a[],us size){
    us i;
    d max = a[0];
    for(i=1;i<size;i++){
        if(a[i] > max) max=a[i];
    }
    return max;
}
/** 
 * Compute the minimum of an array
 *
 * @param a array
 * @param size size of the array
 * @return minimum
 */
static inline d d_min(const d a[],us size){
    us i;
    d min = a[0];
    for(i=1;i<size;i++){
        if(a[i] > min) min=a[i];
    }
    return min;
}

/** 
 * Compute the \f$ L_2 \f$ norm of an array of doubles
 *
 * @param a Array
 * @param size Size of array
 */
static inline d d_norm(const d a[],us size){
    #if LASP_USE_BLAS == 1
    return cblas_dnrm2(size,a,1);
    #else	
    d norm = 0;
    us i;
    for(i=0;i<size;i++){
        norm+=a[i]*a[i];
    }
    norm = d_sqrt(norm);
    return norm;
    #endif
	
}

/** 
 * Compute the \f$ L_2 \f$ norm of an array of complex floats
 *
 * @param a Array
 * @param size Size of array
 */
static inline d c_norm(const c a[],us size){
    #if LASP_USE_BLAS == 1
    return cblas_dznrm2(size,(d*) a,1);
    #else	
    d norm = 0;
    us i;
    for(i=0;i<size;i++){
        d absa = c_abs(a[i]);
        norm+=absa*absa;
    }
    norm = d_sqrt(norm);
    return norm;
    #endif
	
}


/** 
 * Computes the element-wise vector product, or Hadamard product of
 * arr1 and arr2
 *
 * @param res Where the result will be stored
 * @param arr1 Array 1
 * @param vec2 Array 2
 * @param size: Size of the arrays
 */
void d_elem_prod_d(d res[],
                   const d arr1[],
                   const d arr2[],
                   const us size);

/** 
 * Computes the element-wise vector product, or Hadamard product of
 * arr1 and arr2 for complex floats
 *
 * @param res Where the result will be stored
 * @param arr1 Array 1
 * @param vec2 Array 2
 * @param size: Size of the arrays
 */
void c_hadamard(c res[],
                const c arr1[],
                const c arr2[],
                const us size);

/** 
 * Compute the complex conjugate of a complex vector and store the
 * result.
 *
 * @param res Result vector
 * @param in Input vector
 * @param size Size of the vector
 */
static inline void carray_conj(c res[],const c in[],const us size) {
    // First set the result vector to zero
    fsTRACE(15);
    c_set(res,0,size);
    #if LASP_USE_BLAS == 1
    #if LASP_DOUBLE_PRECISION
    // Cast as a float, scale all odd elements with minus one to find
    // the complex conjugate.
    cblas_daxpy(size ,1.0,(d*) in,2,(d*) res,2);
    cblas_daxpy(size,-1.0,&((d*) in)[1],2,&((d*) res)[1],2);
    #else
    cblas_faxpy(size ,1,(d*) in,2,(d*) res,2);
    cblas_faxpy(size,-1,&((d*) in)[1],2,&((d*) res)[1],2);
    #endif  // LASP_DOUBLE_PRECISION
    #else
    for(us i=0;i<size;i++) {
        res[i] = c_conj(in[i]);
    }
    #endif  // LASP_USE_BLAS
    feTRACE(15);
}
/** 
 * In place complex conjugation
 *
 * @param res Result vector
 * @param size Size of the vector
 */
static inline void c_conj_inplace(c res[],us size) {
    #if LASP_USE_BLAS
    #if LASP_DOUBLE_PRECISION
    // Cast as a float, scale all odd elements with minus one to find
    // the complex conjugate.
    cblas_dscal(size,-1,&((d*) res)[1],2);
    #else
    cblas_sscal(size,-1,&((d*) res)[1],2);
    #endif  // LASP_DOUBLE_PRECISION
    #else
    for(us i=0;i<size;i++) {
        res[i] = c_conj(res[i]);
    }
    #endif  // LASP_USE_BLAS
}

#endif // LASP_MATH_RAW_H
//////////////////////////////////////////////////////////////////////
