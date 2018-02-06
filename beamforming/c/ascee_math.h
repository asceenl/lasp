// ascee_math.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Basic routines for allocating, setting, freeing and
// copying of matrices and vectors.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef ASCEE_MATH_H
#define ASCEE_MATH_H
#include "ascee_math_raw.h"
#include "ascee_alloc.h"
#include "ascee_tracer.h"
#include "ascee_assert.h"

/// Vector of floating point numbers
typedef struct {
    us size;
    d* ptr;                     /**< This pointer points to the data
                                   of this vector */
    d* data;                    /**< Pointer set if data storage is
                                   intern. If this is set to zero, the
                                   vector is a sub-vector. */
} vd;
/// Vector of complex floating point numbers
typedef struct {
    us size;
    c* ptr;                     /**< This pointer points to the data
                                   of this vector */
    c* data;                    /**< Pointer set if data storage is
                                   intern. If this is set to zero, the
                                   vector is a sub-vector. */
} vc;
/// Dense matrix of floating point values
typedef struct { 
    us n_rows; 
    us n_cols;
    d** col_ptrs;
    d* data;
} dmat; 
/// Dense matrix of complex floating point values
typedef struct { 
    us n_rows; 
    us n_cols;
    c** col_ptrs;
    c* data; 
} cmat; 

/** 
 * Sets all values in a vector to the value
 *
 * @param vec the vector to set
 * @param value 
 */
static inline void vd_set(vd* vec, d value){
    d_set(vec->ptr,value,vec->size);
}

/** 
 * Sets all values in a vector to the value
 *
 * @param vec the vector to set
 * @param value 
 */
static inline void vc_set(vc* vec,const c value){
    c_set(vec->ptr,value,vec->size);
}

/** 
 * Sets all values in a matrix to the value
 *
 * @param mat The matrix to set
 * @param value 
 */
static inline void dmat_set(dmat* mat,const d value){
    dbgassert(mat,NULLPTRDEREF);
    us size = mat->n_cols*mat->n_rows;
    if(likely(mat->data)){
        d_set(mat->data,value,size);
    }
    else {
        for(us col=0;col<mat->n_cols;col++) {
            d_set(mat->col_ptrs[col],value,mat->n_rows);
        }
    }
}


/** 
 * Sets all values in a matrix to the value
 *
 * @param mat The matrix to set
 * @param value 
 */
static inline void cmat_set(cmat* mat,const c value){
    dbgassert(mat,NULLPTRDEREF);
    us size = mat->n_cols*mat->n_rows;
    if(likely(mat->data)){
        c_set(mat->data,value,size);
    }
    else {
        for(us col=0;col<mat->n_cols;col++) {
            c_set(mat->col_ptrs[col],value,mat->n_rows);
        }
    }
}

/** 
 * Allocate data for a float vector.
 *
 * @param size Size of the vector
 *
 * @return vd with allocated data
 */
static inline vd vd_alloc(us size) {
    vd result = { size, NULL,NULL};
    result.data = (d*) a_malloc(size*sizeof(d));
    result.ptr = result.data;
    dbgassert(result.data,ALLOCFAILED);
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
    vc result = { size, NULL, NULL};
    result.data = (c*) a_malloc(size*sizeof(c));
    result.ptr = result.data;
    #ifdef ASCEE_DEBUG
    vc_set(&result,NAN+I*NAN);
    #endif // ASCEE_DEBUG    
    return result;
}
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
    dmat result = { n_rows, n_cols, NULL, NULL};
    
    /**
     * Here storage is allocated for both the data, as well as the
     * column pointers. The column pointer data is stored at the end
     * of the allocated block.
     */
    result.data = (d*) a_malloc(n_rows*n_cols*sizeof(d)
                                +sizeof(d*)*n_cols);

    dbgassert(result.data,ALLOCFAILED);
    result.col_ptrs = (d**) &result.data[n_rows*n_cols];
    for(us col=0;col<n_cols;col++) {
        result.col_ptrs[col] = &result.data[n_rows*col];
    }
    #ifdef ASCEE_DEBUG
    dmat_set(&result,NAN);
    #endif // ASCEE_DEBUG

    return result;
}


/** 
 * Allocate a matrix of complex floating points
 *
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 * @param p Memory pool
 *
 * @return cmat with allocated data
 */
static inline cmat cmat_alloc(const us n_rows,
                              const us n_cols) {
    cmat result = { n_rows, n_cols, NULL, NULL};
    /**
     * Here storage is allocated for both the data, as well as the
     * column pointers. The column pointer data is stored at the end
     * of the allocated block.
     */
    result.data = (c*) a_malloc(n_rows*n_cols*sizeof(c)
                                +sizeof(c*)*n_cols);

    dbgassert(result.data,ALLOCFAILED);
    result.col_ptrs = (c**) &result.data[n_rows*n_cols];
    for(us col=0;col<n_cols;col++) {
        result.col_ptrs[col] = &result.data[n_rows*col];
    }

    #ifdef ASCEE_DEBUG
    cmat_set(&result,NAN+I*NAN);
    #endif // ASCEE_DEBUG
    return result;
}
/** 
 * Creates a dmat from foreign data. Does not copy the data, but only
 * initializes the row pointers. Assumes column-major ordering for the
 * data. Please do not keep this one alive after the data has been
 * destroyed.
 *
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 * @param data 
 *
 * @return 
 */
static inline dmat dmat_foreign(const us n_rows,
                                const us n_cols,
                                d* data) {

    dbgassert(data,NULLPTRDEREF);
    dmat result = {n_rows,n_cols,NULL,NULL};
    d** colptrs = malloc(sizeof(d*)*n_cols);
    dbgassert(colptrs,ALLOCFAILED);
    result.col_ptrs = colptrs;
    for(us i=0;i<n_cols;i++) {
        colptrs[i] = &data[i*n_rows];
    }
    return result;
}
/** 
 * Creates a cmat from foreign data. Does not copy the data, but only
 * initializes the row pointers. Assumes column-major ordering for the
 * data. Please do not keep this one alive after the data has been
 * destroyed.
 *
 * @param n_rows 
 * @param n_cols 
 * @param data 
 *
 * @return 
 */
static inline cmat cmat_foreign(const us n_rows,
                                const us n_cols,
                                c* data) {
    dbgassert(data,NULLPTRDEREF);
    cmat result = {n_rows,n_cols,NULL,NULL};
    c** colptrs = malloc(sizeof(c*)*n_cols);
    dbgassert(colptrs,ALLOCFAILED);
    result.col_ptrs = colptrs;
    for(us i=0;i<n_cols;i++) {
        colptrs[i] = &data[i*n_rows];
    }
    return result;
}

/** 
 * Free's data of a vector. Is safe to run on sub-vecs as well, to
 * make API consistent. (Only free's data if data pointer is set)
 *
 * @param f Vector to free
 */
static inline void vd_free(vd* f) {
    dbgassert(f,NULLPTRDEREF);
    if(likely(f->data)) a_free(f->data);
}
/** 
 * Free's data of a vector. Is safe to run on sub-vecs as well, to
 * make API consistent. (Only free's data if data pointer is set)
 *
 * @param f Vector to free
 */
static inline void vc_free(vc* f) {
    dbgassert(f,NULLPTRDEREF);
    if(likely(f->data)) a_free(f->data);
}
/** 
 * Free's data of dmat. Safe to run on sub-matrices as well.
 *
 * @param m Matrix to free
 */
static inline void dmat_free(dmat* m) {
    if(likely(m->data)) {
        a_free(m->data);
    }
    else {
        // Only column pointers allocated. This was a submat
        dbgassert(m->col_ptrs,NULLPTRDEREF);
        a_free(m->col_ptrs);
    }
}
/** 
 * Free's data of dmat. Safe to run on sub-matrices as well.
 *
 * @param m Matrix to free
 */
static inline void cmat_free(cmat* m) {
    if(likely(m->data)) {
        a_free(m->data);
    }
    else {
        // Only column pointers allocated. This was a submat
        dbgassert(m->col_ptrs,NULLPTRDEREF);
        a_free(m->col_ptrs);
    }
}

#define setvecval(vec,index,val)                              \
    dbgassert((((us) index) <= (vec)->size),OUTOFBOUNDSVEC);  \
    (vec)->data[index] = val;

    
#define setmatval(mat,row,col,val)                              \
    dbgassert((((us) row) <= mat->n_rows),OUTOFBOUNDSMATR);     \
    dbgassert((((us) col) <= mat->n_cols),,OUTOFBOUNDSMATC);    \
    (mat)->data[(col)*(mat)->n_rows+(row)] = val;

/** 
 * Return pointer to a value from a vector
 *
 * @param mat The vector
 * @param row The row
 */
static inline d* getvdval(const vd* vec,us row){
    dbgassert(row < vec->size,OUTOFBOUNDSVEC);
    return &vec->ptr[row];
}

/** 
 * Return pointer to a value from a complex vector
 *
 * @param mat The vector
 * @param row The row
 */
static inline c* getvcval(const vc* vec,us row){
    dbgassert(row < vec->size,OUTOFBOUNDSVEC);
    return &vec->ptr[row];
}

/** 
 * Return a value from a matrix of floating points
 *
 * @param mat The matrix
 * @param row The row
 * @param col The column
 */
static inline d* getdmatval(const dmat* mat,us row,us col){
    dbgassert(mat,NULLPTRDEREF);
    dbgassert(row < mat->n_rows,OUTOFBOUNDSMATR);
    dbgassert(col < mat->n_cols,OUTOFBOUNDSMATC);
    return &mat->col_ptrs[col][row];
}

/** 
 * Return a value from a matrix of complex floating points
 *
 * @param mat The matrix
 * @param row The row
 * @param col The column
 */
static inline c* getcmatval(const cmat* mat,const us row,const us col){
    dbgassert(mat,NULLPTRDEREF);
    dbgassert(row < mat->n_rows,OUTOFBOUNDSMATR);
    dbgassert(col < mat->n_cols,OUTOFBOUNDSMATC);
    return &mat->col_ptrs[col][row];
}

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
 * Allocate a sub-matrix view of the parent
 *
 * @param parent Parent matrix
 * @param startrow Startrow
 * @param startcol Start column
 * @param n_rows Number of rows in sub-matrix
 * @param n_cols Number of columns in sub-matrix
 *
 * @return submatrix view
 */
static inline dmat dmat_submat(const dmat* parent,
                               const us startrow,
                               const us startcol,
                               const us n_rows,
                               const us n_cols) {

    dbgassert(parent,NULLPTRDEREF);
    dbgassert(n_rows+startrow <= parent->n_rows,OUTOFBOUNDSMATR);
    dbgassert(n_cols+startcol <= parent->n_cols,OUTOFBOUNDSMATC);

    d** col_ptrs = malloc(sizeof(d*)*n_cols);
    dbgassert(col_ptrs,ALLOCFAILED);
    for(us col=0;col<n_cols;col++) {
        col_ptrs[col] = getdmatval(parent,
                                   startrow,
                                   startcol+col);

    }
    dmat result = { n_rows,n_cols,col_ptrs,NULL};
    return result;
}
/** 
 * Allocate a sub-matrix view of the parent
 *
 * @param parent Parent matrix
 * @param startrow Startrow
 * @param startcol Start column
 * @param n_rows Number of rows in sub-matrix
 * @param n_cols Number of columns in sub-matrix
 *
 * @return submatrix view
 */
static inline cmat cmat_submat(cmat* parent,
                               const us startrow,
                               const us startcol,
                               const us n_rows,
                               const us n_cols) {

    dbgassert(parent,NULLPTRDEREF);
    dbgassert(n_rows+startrow <= parent->n_rows,OUTOFBOUNDSMATR);
    dbgassert(n_cols+startcol <= parent->n_cols,OUTOFBOUNDSMATC);

    c** col_ptrs = malloc(sizeof(c*)*n_cols);
    dbgassert(col_ptrs,ALLOCFAILED);
    for(us col=0;col<n_cols;col++) {
        col_ptrs[col] = getcmatval(parent,
                                   startrow,
                                   startcol+col);

    }
    cmat result = { n_rows,n_cols,col_ptrs,NULL};
    return result;
}

/** 
 * Copy contents of one vector to another
 *
 * @param to : Vector to write to
 * @param from : Vector to read from
 */
static inline void vd_copy(vd* to,vd* from) {
    dbgassert(to && from,NULLPTRDEREF);
    dbgassert(to->size==from->size,SIZEINEQUAL);
    d_copy(to->ptr,from->ptr,to->size);
}
/** 
 * Copy contents of one vector to another
 *
 * @param to : Vector to write to
 * @param from : Vector to read from
 */
static inline void vc_copy(vc* to,vc* from) {
    dbgassert(to && from,NULLPTRDEREF);
    dbgassert(to->size==from->size,SIZEINEQUAL);
    c_copy(to->ptr,from->ptr,to->size);
}
/** 
 * Copy contents of one matrix to another. Sizes should be equal
 *
 * @param to 
 * @param from 
 */
static inline void dmat_copy(dmat* to,const dmat* from) {
    dbgassert(to && from,NULLPTRDEREF);
    dbgassert(to->n_rows==from->n_rows,SIZEINEQUAL);
    dbgassert(to->n_cols==from->n_cols,SIZEINEQUAL);    
    for(us col=0;col<to->n_cols;col++) {
        d_copy(to->col_ptrs[col],from->col_ptrs[col],to->n_rows);
    }
}
/** 
 * Copy contents of one matrix to another. Sizes should be equal
 *
 * @param to 
 * @param from 
 */
static inline void cmat_copy(cmat* to,const cmat* from) {
    dbgassert(to && from,NULLPTRDEREF);
    dbgassert(to->n_rows==from->n_rows,SIZEINEQUAL);
    dbgassert(to->n_cols==from->n_cols,SIZEINEQUAL);    
    for(us col=0;col<to->n_cols;col++) {
        c_copy(to->col_ptrs[col],from->col_ptrs[col],to->n_rows);
    }
}


/** 
 * Get a reference to a column of a matrix as a vector
 *
 * @param x Matrix
 * @param col Column number
 *
 * @return vector with reference to column
 */
static inline vd dmat_column(dmat* x,us col) {
    vd res = { x->n_rows, getdmatval(x,0,col),NULL};
    return res;
}

/** 
 * Get a reference to a column of a matrix as a vector
 *
 * @param x Matrix
 * @param col Column number
 *
 * @return vector with reference to column
 */
static inline vc cmat_column(cmat* x,us col) {
    vc res = { x->n_rows, getcmatval(x,0,col),NULL};
    return res;
}

/** 
 * Compute the complex conjugate of b and store result in a
 *
 * @param a 
 * @param b 
 */
static inline void vc_conj_vc(vc* a,const vc* b) {
    dbgassert(a && b,NULLPTRDEREF);
    dbgassert(a->size == b->size,SIZEINEQUAL);
    c_conj_c(a->ptr,b->ptr,a->size);
}

/** 
 * Take the complex conjugate of x, in place
 *
 * @param x 
 */
static inline void cmat_conj(cmat* x) {
    dbgassert(x,NULLPTRDEREF);
    if(likely(x->data)) {
        c_conj_inplace(x->data,x->n_cols*x->n_rows);
    }
    else {
        for(us col=0;col<x->n_cols;col++) {
            c_conj_inplace(x->col_ptrs[col],x->n_rows);
        }
    }
}


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

#endif // ASCEE_MATH_H
//////////////////////////////////////////////////////////////////////
