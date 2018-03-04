// lasp_math.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Basic routines for allocating, setting, freeing and
// copying of matrices and vectors.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_MATH_H
#define LASP_MATH_H
#include "lasp_math_raw.h"
#include "lasp_alloc.h"
#include "lasp_tracer.h"
#include "lasp_assert.h"

/// Vector of floating point numbers
typedef struct {
    us size;
    bool _foreign_data;
    d* _data;                    /**< Pointer set if data storage is
                                   intern. If this is set to zero, the
                                   vector is a sub-vector. */
} vd;
/// Vector of complex floating point numbers
typedef struct {
    us size;
    bool _foreign_data;
    c* _data;                    /**< Pointer set if data storage is
                                   intern. If this is set to zero, the
                                   vector is a sub-vector. */
} vc;
/// Dense matrix of floating point values
typedef struct { 
    us n_rows; 
    us n_cols;
    bool _foreign_data;
    us stride;
    d* _data;
} dmat; 
/// Dense matrix of complex floating point values
typedef struct { 
    us n_rows; 
    us n_cols;
    bool _foreign_data;
    us stride;
    c* _data; 
} cmat; 

#define setvecval(vec,index,val)                              \
    dbgassert((((us) index) <= (vec)->size),OUTOFBOUNDSVEC);  \
    (vec)->_data[index] = val;

    
#define setmatval(mat,row,col,val)                              \
    dbgassert((((us) row) <= mat->n_rows),OUTOFBOUNDSMATR);     \
    dbgassert((((us) col) <= mat->n_cols),,OUTOFBOUNDSMATC);    \
    (mat)->data[(col)*(mat)->stride+(row)] = val;

/** 
 * Return pointer to a value from a vector
 *
 * @param mat The vector
 * @param row The row
 */
static inline d* getvdval(const vd* vec,us row){
    dbgassert(row < vec->size,OUTOFBOUNDSVEC);
    return &vec->_data[row];
}

/** 
 * Return pointer to a value from a complex vector
 *
 * @param mat The vector
 * @param row The row
 */
static inline c* getvcval(const vc* vec,us row){
    dbgassert(row < vec->size,OUTOFBOUNDSVEC);
    return &vec->_data[row];
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
    return &mat->_data[col*mat->stride+row];
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
    return &mat->_data[col*mat->stride+row];
}

#ifdef LASP_DEBUG
#define OVERFLOW_MAGIC_NUMBER (-10e-45)

#define check_overflow_vx(vx)                   \
    TRACE(15,"Checking overflow " #vx);          \
    if(!(vx)._foreign_data) {                                            \
        dbgassert((vx)._data[(vx).size] == OVERFLOW_MAGIC_NUMBER,       \
                  "Buffer overflow detected on" #vx );          \
    }                                                           \
    else {                                                      \
    DBGWARN("Cannot check overflow on foreign buffer");            \
    }

#define check_overflow_xmat(xmat)                               \
    TRACE(15,"Checking overflow " #xmat);                        \
    if(!(xmat)._foreign_data) {                                  \
        dbgassert((xmat)._data[((xmat).n_cols-1)*(xmat).stride+(xmat).n_rows] \
                  == OVERFLOW_MAGIC_NUMBER,                             \
                  "Buffer overflow detected on" #xmat );                \
    }                                                                   \
    else {                                                      \
    DBGWARN("Cannot check overflow on foreign buffer");            \
    }

#else
#define check_overflow_vx(vx)
#define check_overflow_xmat(xmat)
#endif

/** 
 * Sets all values in a vector to the value
 *
 * @param vec the vector to set
 * @param value 
 */
static inline void vd_set(vd* vec, d value){
    d_set(vec->_data,value,vec->size);
}

/** 
 * Sets all values in a vector to the value
 *
 * @param vec the vector to set
 * @param value 
 */
static inline void vc_set(vc* vec,const c value){
    c_set(vec->_data,value,vec->size);
}

/** 
 * Sets all values in a matrix to the value
 *
 * @param mat The matrix to set
 * @param value 
 */
static inline void dmat_set(dmat* mat,const d value){
    dbgassert(mat,NULLPTRDEREF);
    if(likely(mat->n_cols && mat->n_rows)) {
        for(us col=0;col<mat->n_cols;col++) {
            d_set(getdmatval(mat,0,col),value,mat->n_rows);
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
    if(likely(mat->n_cols && mat->n_rows)) {
        for(us col=0;col<mat->n_cols;col++) {
            c_set(getcmatval(mat,0,col),value,mat->n_rows);
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
    #ifdef LASP_DEBUG
    result._data = (d*) a_malloc((size+1)*sizeof(d));
    result._data[size] = OVERFLOW_MAGIC_NUMBER;
    #else
    result._data = (d*) a_malloc(size*sizeof(d));    
    #endif //  LASP_DEBUG
    result._foreign_data = false;
    #ifdef LASP_DEBUG
    vd_set(&result,NAN);
    #endif // LASP_DEBUG    
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
    #ifdef LASP_DEBUG
    result._data = (c*) a_malloc((size+1)*sizeof(c));    
    result._data[size] = OVERFLOW_MAGIC_NUMBER;
    #else
    result._data = (c*) a_malloc(size*sizeof(c));    
    #endif //  LASP_DEBUG
    result._foreign_data = false;
    #ifdef LASP_DEBUG
    vc_set(&result,NAN+I*NAN);
    #endif // LASP_DEBUG    
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
    dmat result = { n_rows, n_cols, false, n_rows, NULL};
    
    #ifdef LASP_DEBUG
    result._data = (d*) a_malloc((n_rows*n_cols+1)*sizeof(d));
    result._data[n_rows*n_cols] = OVERFLOW_MAGIC_NUMBER;
    #else
    result._data = (d*) a_malloc((n_rows*n_cols)*sizeof(d));    
    #endif //  LASP_DEBUG

    #ifdef LASP_DEBUG
    dmat_set(&result,NAN);
    #endif // LASP_DEBUG

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
    cmat result = { n_rows, n_cols, false, n_rows, NULL};

    #ifdef LASP_DEBUG
    result._data = (c*) a_malloc((n_rows*n_cols+1)*sizeof(c));
    result._data[n_rows*n_cols] = OVERFLOW_MAGIC_NUMBER;
    #else
    result._data = (c*) a_malloc((n_rows*n_cols)*sizeof(c));    
    #endif //  LASP_DEBUG

    #ifdef LASP_DEBUG
    cmat_set(&result,NAN+I*NAN);
    #endif // LASP_DEBUG
    return result;
}
/** 
 * Creates a dmat from foreign data. Does not copy the data, but only
 * initializes the row pointers. Assumes column-major ordering for the
 * data. Please do not keep this one alive after the data has been
 * destroyed. Assumes the column stride equals to n_rows.
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
    dmat result = {n_rows,n_cols,true,n_rows,data};
    return result;
}
static inline dmat dmat_foreign_vd(vd* vector) {
    dbgassert(vector,NULLPTRDEREF);
    dmat result = {vector->size,1,true,vector->size,vector->_data};
    return result;
}
/** 
 * Create vd from foreign data
 *
 * @param size Size of the vector
 * @param data Pointer to data
 *
 * @return 
 */
static inline vd vd_foreign(const us size,d* data) {
    dbgassert(data,NULLPTRDEREF);
    vd result = {size,true,data};
    return result;
}
/** 
 * Creates a cmat from foreign data. Does not copy the data, but only
 * initializes the row pointers. Assumes column-major ordering for the
 * data. Please do not keep this one alive after the data has been
 * destroyed. Assumes the column stride equals to n_rows.
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
    cmat result = {n_rows,n_cols,true,n_rows,data};
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
    if(!(f->_foreign_data)) a_free(f->_data);
}
/** 
 * Free's data of a vector. Is safe to run on sub-vecs as well, to
 * make API consistent. (Only free's data if data pointer is set)
 *
 * @param f Vector to free
 */
static inline void vc_free(vc* f) {
    dbgassert(f,NULLPTRDEREF);
    if(!(f->_foreign_data)) a_free(f->_data);
}
/** 
 * Free's data of dmat. Safe to run on sub-matrices as well.
 *
 * @param m Matrix to free
 */
static inline void dmat_free(dmat* m) {
    dbgassert(m,NULLPTRDEREF);
    if(!(m->_foreign_data)) a_free(m->_data);
}
/** 
 * Free's data of dmat. Safe to run on sub-matrices as well.
 *
 * @param m Matrix to free
 */
static inline void cmat_free(cmat* m) {
    dbgassert(m,NULLPTRDEREF);
    if(!(m->_foreign_data)) a_free(m->_data);
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
static inline void dmat_copy_rows(dmat* to,const dmat* from,
                                  us startrow_to,
                                  us startrow_from,
                                  us nrows) {
    us col,ncols = to->n_cols;

    dbgassert(startrow_from+nrows <= from->n_rows,OUTOFBOUNDSMATR);
    dbgassert(startrow_to+nrows <= to->n_rows,OUTOFBOUNDSMATR);
    for(col=0;col<ncols;col++) {
        d* to_d = getdmatval(to,startrow_to,col);
        d* from_d = getdmatval(from,startrow_from,col);
        d_copy(to_d,from_d,nrows,1,1);
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

    dmat result = { n_rows,n_cols,
                    true,           // Foreign data = true
                    parent->n_rows, // This is the stride to get to
                    // the next column.
                    getdmatval(parent,startrow,startcol)};

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
    dbgassert(false,"untested");
    dbgassert(parent,NULLPTRDEREF);
    dbgassert(n_rows+startrow <= parent->n_rows,OUTOFBOUNDSMATR);
    dbgassert(n_cols+startcol <= parent->n_cols,OUTOFBOUNDSMATC);


    cmat result = { n_rows,n_cols,
                    true,           // Foreign data = true
                    parent->n_rows, // This is the stride to get to
                    // the next column.
                    getcmatval(parent,startrow,startcol)};

    return result;
}

/** 
 * Copy contents of one vector to another
 *
 * @param to : Vector to write to
 * @param from : Vector to read from
 */
static inline void vd_copy(vd* to,const vd* from) {
    dbgassert(to && from,NULLPTRDEREF);
    dbgassert(to->size==from->size,SIZEINEQUAL);
    d_copy(to->_data,from->_data,to->size,1,1);
}
static inline void vd_copy_rows(vd* to,
                                const vd* from,
                                const us startrow_to,
                                const us startrow_from,
                                const us nrows) {
    dbgassert(to && from,NULLPTRDEREF);
    dbgassert(startrow_from+nrows <= from->size,OUTOFBOUNDSMATR);    
    dbgassert(startrow_to+nrows <= to->size,OUTOFBOUNDSMATR);
    d_copy(&to->_data[startrow_to],
           &from->_data[startrow_from],
           nrows,1,1);
}
/** 
 * Copy contents of one vector to another
 *
 * @param to : Vector to write to
 * @param from : Vector to read from
 */
static inline void vc_copy(vc* to,const vc* from) {
    dbgassert(to && from,NULLPTRDEREF);
    dbgassert(to->size==from->size,SIZEINEQUAL);
    c_copy(to->_data,from->_data,to->size);
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
        d_copy(getdmatval(to,0,col),
               getdmatval(from,0,col),
               to->n_rows,1,1);
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
        c_copy(getcmatval(to,0,col),
               getcmatval(from,0,col),
               to->n_rows);
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
    vd res = {x->n_rows,true,getdmatval(x,0,col)};
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
    vc res = {x->n_rows,true,getcmatval(x,0,col)};
    return res;
}

/** 
 * Compute the complex conjugate of b and store result in a
 *
 * @param a 
 * @param b 
 */
static inline void vc_conj(vc* a,const vc* b) {
    fsTRACE(15);
    dbgassert(a && b,NULLPTRDEREF);
    dbgassert(a->size == b->size,SIZEINEQUAL);
    carray_conj(a->_data,b->_data,a->size);
    feTRACE(15);
}

/** 
 * Take the complex conjugate of x, in place
 *
 * @param x 
 */
static inline void cmat_conj(cmat* x) {
    dbgassert(x,NULLPTRDEREF);
    for(us col=0;col<x->n_cols;col++) {
        c_conj_inplace(getcmatval(x,0,col),x->n_rows);
    }
}


#ifdef LASP_DEBUG
void print_cmat(const cmat* m);
void print_vc(const vc* m);
void print_vd(const vd* m);
void print_dmat(const dmat* m);
#else
#define print_cmat(m)
#define print_vc(m)
#define print_dmat(m)
#endif

#endif // LASP_MATH_H
//////////////////////////////////////////////////////////////////////
