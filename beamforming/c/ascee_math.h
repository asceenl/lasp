// ascee_math.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef ASCEE_MATH_H
#define ASCEE_MATH_H
#include <math.h>
#include "types.h"
#include "tracer.h"
#include "ascee_assert.h"
#include "ascee_alloc.h"

#if ASCEE_USE_BLAS == 1
#include <cblas.h>
#endif

#ifdef ASCEE_DOUBLE_PRECISION
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

#else  // ASCEE_DOUBLE_PRECISION not defined
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

#endif // ASCEE_DOUBLE_PRECISION

#ifdef M_PI
static const d number_pi = M_PI;
#else
static const d number_pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;
#endif


/// Code generation for vector of floats and vector of complex floats.
#define vxinit(type)                                               \
    typedef struct {                                            \
        us size;                                                \
        type* data;                                             \
    } v##type;
vxinit(d);
vxinit(c);

/// Code generation for matrix of floats and matrix of complex floats.
#define xmatinit(type)                                                  \
    typedef struct {                                                 \
        us n_rows;                                                    \
        us n_cols;                                                    \
        type* data;                                                     \
    } type##mat;
xmatinit(d);
xmatinit(c);


    
#define setvecval(vec,index,val)                              \
    dbgassert((((us) index) <= (vec)->size),OUTOFBOUNDSVEC);  \
    (vec)->data[index] = val;

    
#define setmatval(mat,row,col,val)                              \
    dbgassert((((us) row) <= mat->n_rows),OUTOFBOUNDSMATR);     \
    dbgassert((((us) col) <= mat->n_cols),,OUTOFBOUNDSMATC);    \
    (mat)->data[(col)*(mat)->n_rows+(row)] = val;

/** 
 * Return a value from a vector
 *
 * @param mat The vector
 * @param row The row
 */
static inline d* getvdval(const vd* vec,us row){
    dbgassert(row < vec->size,OUTOFBOUNDSVEC);
    return &vec->data[row];
}

/** 
 * Return a value from a complex vector
 *
 * @param mat The vector
 * @param row The row
 */
static inline c* getvcval(const vc* vec,us row){
    dbgassert(row < vec->size,OUTOFBOUNDSVEC);
    return &vec->data[row];
}

/** 
 * Return a value from a matrix of floating points
 *
 * @param mat The matrix
 * @param row The row
 * @param col The column
 */
static inline d* getdmatval(const dmat* mat,us row,us col){
    assert((row) < mat->n_rows); 
    assert((col) < mat->n_cols);
    return &mat->data[(col)*mat->n_rows+(row)];
}

/** 
 * Return a value from a matrix of complex floating points
 *
 * @param mat The matrix
 * @param row The row
 * @param col The column
 */
static inline c* getcmatval(const cmat* mat,const us row,const us col){
    dbgassert(row < mat->n_rows,OUTOFBOUNDSMATR);
    dbgassert(col < mat->n_cols,OUTOFBOUNDSMATC);
    return &mat->data[col*mat->n_rows+row];
}

/** 
 * Sets all values in a vector to the value
 *
 * @param b the vector to set
 * @param value 
 */
static inline void vd_set(vd* vec, d value){
    us i;
    for(i=0;i<vec->size;i++){
        vec->data[i] = value;
    }
}

/** 
 * Sets all values in a vector to the value
 *
 * @param vec the vector to set
 * @param value 
 */
static inline void vc_set(vc* vec,const c value){
    us i;
    for(i=0;i<vec->size;i++){
        vec->data[i] = value;
    }
}

/** 
 * Sets all values in a matrix to the value
 *
 * @param mat The matrix to set
 * @param value 
 */
static inline void dmat_set(dmat* mat,const d value){
    us i,size = mat->n_cols*mat->n_rows;
    for(i=0;i<size;i++){
        mat->data[i] = value;
    }
}


/** 
 * Sets all values in a matrix to the value
 *
 * @param mat The matrix to set
 * @param value 
 */
static inline void cmat_set(cmat* mat,const c value){
    us i,size = mat->n_cols*mat->n_rows;
    for(i=0;i<size;i++){
        mat->data[i] = value;
    }
}

/** 
 * Return a column pointer of the matrix
 *
 * @param mtrx The matrix.
 * @param column The column number.
 *
 * @return Pointer to the column.
 */
static inline d* d_column(dmat* mtrx,us column){
    return &mtrx->data[mtrx->n_rows*column];
}

/** 
 * Return a column pointer of the matrix
 *
 * @param mtrx The matrix.
 * @param column The column number.
 *
 * @return Pointer to the column.
 */
static inline c* c_column(cmat* mtrx,us column){
    return &mtrx->data[mtrx->n_rows*column];
}


/** 
 * Return the maximum of two doubles
 *
 * @param a value 1
 * @param b value 2
 * 
 * @returns the maximum of value 1 and 2
 */
static inline d max(const d a,const d b) {
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
 * when ASCEE_USE_BLAS == 1.
 *
 * @param a complex-valued array
 * @param b complex-valued array
 * @param size the size of the arrays. *Should be equal-sized!*
 *
 * @return the dot product
 */
static inline c cc_dot(const c a[],const c b[],us size){
    #if ASCEE_USE_BLAS == 1
    WARN("CBlas zdotu not yet tested");
    #if ASCEE_DOUBLE_PRECISION
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
    #if ASCEE_USE_BLAS == 1
    #if ASCEE_DOUBLE_PRECISION
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
 * Compute the dot product of two vectors of double precision
 *
 * @param a First vector
 * @param b Second second vector

 */
static inline d vd_dot(const vd * a,const vd* b) {
    dbgassert(a->size == b->size,SIZEINEQUAL);
    return d_dot(a->data,b->data,a->size);
}


/** 
 * Copy array of floats.
 *
 * @param to : Array to write to
 * @param from : Array to read from
 * @param size : Size of arrays
 */
static inline void d_copy(d to[],const d from[],const us size){
    #if ASCEE_USE_BLAS == 1
    cblas_dcopy(size,from,1,to,1);
    #else
    us i;
    for(i=0;i<size;i++)
        to[i] = from[i];
    #endif
}

/** 
 * Copy vector to another
 *
 * @param to : Vector to write to
 * @param from : Vector to read from
 */
static inline void vd_copy(vd* to,vd* from) {
    dbgassert(to->size==from->size,SIZEINEQUAL);
    d_copy(to->data,from->data,to->size);
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
static inline void c_copy(c to[],const c from[],us size){
    
    #if ASCEE_USE_BLAS == 1
    #if ASCEE_DOUBLE_PRECISION
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
 * Add a constant factor 'fac' to elements of y, and write result to
 * x.
 *
 * @param x Array to add to
 * @param y Array to add to x
 * @param fac Factor with which to multiply y
 * @param size Size of the arrays
 */
static inline void d_add_to(d x[],const d y[],d fac,us size){
    #if ASCEE_USE_BLAS == 1
    #if ASCEE_DOUBLE_PRECISION
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
 * Scale an array of doubles
 *
 * @param a array
 * @param scale_fac scale factor
 * @param size size of the array
 */
static inline void d_scale(d a[],const d scale_fac,us size){
    #if ASCEE_USE_BLAS == 1
    #if ASCEE_DOUBLE_PRECISION    
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
    #if ASCEE_USE_BLAS == 1
    // Complex argument should be given in as array of two double
    // values. The first the real part, the second the imaginary
    // part. Fortunately the (c) type stores the two values in this
    // order. To be portable and absolutely sure anything goes well,
    // we convert it explicitly here.
    d scale_fac_d [] = {creal(scale_fac),cimag(scale_fac)};

    #if ASCEE_DOUBLE_PRECISION
    cblas_zscal(size,scale_fac_d,(d*) a,1);
    #else
    cblas_cscal(size,scale_fac_d,(d*) a,1);
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
static inline d d_max(const d a[],us size){
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
    #if ASCEE_USE_BLAS == 1
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
    #if ASCEE_USE_BLAS == 1
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
 * Computes the Kronecker product of a kron b, stores result in result.
 *
 * @param a a
 * @param b b
 * @param result a kron b
 */
void kronecker_product(const cmat* a,const cmat* b,cmat* result);

#ifdef ASCEE_DEBUG
void print_cmat(const cmat* m);
void print_vc(const vc* m);
void print_vd(const vd* m);
void print_dmat(const dmat* m);
#else
#define print_cmat(m)
#define print_vc(m)
#define print_dmat(m)
#endif


/** 
 * Allocate data for a float vector.
 *
 * @param size Size of the vector
 *
 * @return vd with allocated data
 */
static inline vd vd_alloc(us size) {
    vd result = { size, NULL};
    result.data = (d*) a_malloc(size*sizeof(d));
    #ifdef ASCEE_DEBUG
    vd_set(&result,NAN);
    #endif // ASCEE_DEBUG    
    return result;
}
/** 
 * Allocate data for a complex vector.
 *
 * @param size Size of the vector
 *
 * @return vc with allocated data
 */
static inline vc vc_alloc(us size) {
    vc result = { size, NULL};
    result.data = (c*) a_malloc(size*sizeof(c));
    #ifdef ASCEE_DEBUG
    vc_set(&result,NAN+I*NAN);
    #endif // ASCEE_DEBUG    
    return result;
}

/**
 * Free the data of a dmat, cmat, vd, or vc. This function is
 * macro-nized as what is to be done is the same for each of these
 * types, free-ing the buffer.
 */
#define matvec_free(type)                       \
    static inline void type##_free(type * buf) {       \
    a_free(buf->data);                                  \
    }
matvec_free(vd);
matvec_free(vc);
matvec_free(dmat);
matvec_free(cmat);

/**
 * Now the following functions exist: vd_free, vc_free, dmat_free and
 * cmat_free.
 */


/** 
 * Allocate data for a matrix of floating points
 *
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 * @param p Memory pool
 *
 * @return dmat with allocated data
 */
static inline dmat dmat_alloc(us n_rows,
                              us n_cols) {
    dmat result = { n_rows, n_cols, NULL};
    result.data = (d*) a_malloc(n_rows*n_cols*sizeof(d));
    #ifdef ASCEE_DEBUG
    dmat_set(&result,NAN);
    #endif // ASCEE_DEBUG
    assert(result.data);
    return result;
}


/** 
 * Allocate data for a matrix of complex floating points
 *
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 * @param p Memory pool
 *
 * @return cmat with allocated data
 */
static inline cmat cmat_alloc(us n_rows,
                              us n_cols) {
    cmat result = { n_rows, n_cols, NULL};
    result.data = (c*) a_malloc(n_rows*n_cols*sizeof(c));
    #ifdef ASCEE_DEBUG
    cmat_set(&result,NAN+I*NAN);
    #endif // ASCEE_DEBUG
    assert(result.data);
    return result;
}

/**
 * Resize an existing dmat or a cmat 
 */
#define type_mat_resize(type)       \
    static inline void type##mat_resize(type##mat * mat,\
                                        us nrows,us ncols) {    \
    mat->n_rows = nrows;                                        \
    mat->n_cols = ncols;                                        \
    mat->data = realloc(mat->data,(nrows*ncols)*sizeof( type ));        \
    }
type_mat_resize(d);
type_mat_resize(c);

/** 
 * Copy some rows from one matrix to another
 *
 * @param to Matrix to copy to
 * @param from Matrix to copy from
 * @param startrow_from Starting row where to get the values
 * @param startrow_to Starting row where to insert the values
 * @param nrows Number of rows to copy
 */
static inline void copy_dmat_rows(dmat* to,const dmat* from,
                                  us startrow_from,
                                  us startrow_to,
                                  us nrows) {
    us col,ncols = to->n_cols;

    dbgassert(startrow_from+nrows <= from->n_rows,OUTOFBOUNDSMATR);
    dbgassert(startrow_to+nrows <= to->n_rows,OUTOFBOUNDSMATR);
    for(col=0;col<ncols;col++) {
        d* to_d = getdmatval(to,startrow_to,col);
        d* from_d = getdmatval(from,startrow_from,col);
        d_copy(to_d,from_d,nrows);
    }

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
void c_elem_prod_c(c res[],
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
static inline void c_conj_c(c res[],const c in[],us size) {
    for(us i=0;i<size;i++) {
        res[i] = c_conj(in[i]);
    }
}
/** 
 * In place complex conjugation
 *
 * @param res Result vector
 * @param size Size of the vector
 */
static inline void c_conj_inplace(c res[],us size) {
    for(us i=0;i<size;i++) {
        res[i] = c_conj(res[i]);
    }
}


/** 
 * Compute the matrix vector product for complex-valued types: b = A*x.
 *
 * @param[in] A Matrix A
 * @param[in] x Vector x
 * @param[out] b Result of computation
 */
void cmv_dot(const cmat* A,
             const vc* restrict x,
             vc* restrict b);

int lsq_solve(const cmat* A,
              const vc* restrict b,
              vc* restrict x);

// Compute the Frobenius norm of A-B
d c_normdiff(const cmat* A,const cmat* B);


#endif // SI_MATH_H
//////////////////////////////////////////////////////////////////////
