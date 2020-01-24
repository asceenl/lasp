// lasp_mat.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: Basic routines for allocating, setting, freeing and
// copying of matrices and vectors.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_MAT_H
#define LASP_MAT_H
#include "lasp_math_raw.h"
#include "lasp_alloc.h"
#include "lasp_assert.h"
#include "lasp_tracer.h"
#include "lasp_assert.h"

/// Dense matrix of floating point values
typedef struct {
    us n_rows;
    us n_cols;
    bool _foreign_data;
    us colstride;
    d* _data;
} dmat;

/// Dense matrix of complex floating point values
typedef struct {
    us n_rows;
    us n_cols;
    bool _foreign_data;
    us colstride;
    c* _data;
} cmat;

typedef dmat vd;
typedef cmat vc;

#define assert_equalsize(a,b)                       \
    dbgassert((a)->n_rows == (b)->n_rows,SIZEINEQUAL);  \
    dbgassert((a)->n_cols == (b)->n_cols,SIZEINEQUAL);

#define is_vx(vx) ((vx)->n_cols == 1)
#define assert_vx(vx) dbgassert(is_vx(vx),"Not a vector!")

#define setvecval(vec,index,val)                              \
    assert_vx(vec);                                           \
    dbgassert((((us) index) < (vec)->n_rows),OUTOFBOUNDSVEC);  \
    (vec)->_data[index] = val;

#define setmatval(mat,row,col,val)                              \
    dbgassert((((us) row) <= (mat)->n_rows),OUTOFBOUNDSMATR);     \
    dbgassert((((us) col) <= (mat)->n_cols),OUTOFBOUNDSMATC);    \
    (mat)->_data[(col)*(mat)->colstride+(row)] = val;

/**
 * Return pointer to a value from a vector
 *
 * @param mat The vector
 * @param row The row
 */
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
    return &mat->_data[col*mat->colstride+row];
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
    return &mat->_data[col*mat->colstride+row];
}

static inline d* getvdval(const vd* vec,us row){
    dbgassert(vec,NULLPTRDEREF);
    assert_vx(vec);
    return getdmatval(vec,row,0);
}

/**
 * Return pointer to a value from a complex vector
 *
 * @param mat The vector
 * @param row The row
 */
static inline c* getvcval(const vc* vec,us row){
    dbgassert(vec,NULLPTRDEREF);
    assert_vx(vec);
    return getcmatval(vec,row,0);
}



#ifdef LASP_DEBUG
#define OVERFLOW_MAGIC_NUMBER (-10e-45)

#define check_overflow_xmat(xmat)                               \
    TRACE(15,"Checking overflow " #xmat);                        \
    if(!(xmat)._foreign_data) {                                  \
        dbgassert((xmat)._data[((xmat).n_cols-1)*(xmat).colstride+(xmat).n_rows] \
                  == OVERFLOW_MAGIC_NUMBER,                             \
                  "Buffer overflow detected on" #xmat );                \
    }                                                                   \

#define check_overflow_vx check_overflow_xmat

#else
#define check_overflow_vx(vx)
#define check_overflow_xmat(xmat)
#endif

/**
 * Sets all values in a matrix to the value
 *
 * @param mat The matrix to set
 * @param value
 */
static inline void dmat_set(dmat* mat,const d value){
    dbgassert(mat,NULLPTRDEREF);
    if(likely(mat->n_cols * mat->n_rows > 0)) {
        for(us col=0;col<mat->n_cols;col++) {
            d_set(getdmatval(mat,0,col),value,mat->n_rows);
        }
    }
}
#define vd_set dmat_set

/**
 * Sets all values in a matrix to the value
 *
 * @param mat The matrix to set
 * @param value
 */
static inline void cmat_set(cmat* mat,const c value){
    dbgassert(mat,NULLPTRDEREF);
    if(likely(mat->n_cols * mat->n_rows > 0)) {
        for(us col=0;col<mat->n_cols;col++) {
            c_set(getcmatval(mat,0,col),value,mat->n_rows);
        }
    }
}
#define vc_set cmat_set

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
    dmat result = { n_rows, n_cols, 
                   false,
                   n_rows, // The column stride
                   NULL};

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
 * Allocate data for a float vector.
 *
 * @param size Size of the vector
 *
 * @return vd with allocated data
 */
static inline vd vd_alloc(us size) {
    return dmat_alloc(size,1);
}

/**
 * Allocate data for a complex vector.
 *
 * @param size Size of the vector
 *
 * @return vc with allocated data
 */
static inline vc vc_alloc(us size) {
    return cmat_alloc(size,1);
}


/**
 * Creates a dmat from foreign data. Does not copy the data, but only
 * initializes the row pointers. Assumes column-major ordering for the
 * data. Please do not keep this one alive after the data has been
 * destroyed. Assumes the column colstride equals to n_rows.
 *
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 * @param data
 *
 * @return
 */
static inline dmat dmat_foreign(dmat* other) {
    dbgassert(other,NULLPTRDEREF);
    dmat result = {other->n_rows,
                   other->n_cols,
                   true,
                   other->colstride,
                   other->_data};
    return result;
}
/**
 * Create a dmat from foreign data. Assumes the colstride of the data is
 * n_rows.
 *
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 * @param data Pointer to data storage
 *
 * @return dmat
 */
static inline dmat dmat_foreign_data(us n_rows,
                                     us n_cols,
                                     d* data,
                                     bool own_data) {

    dbgassert(data,NULLPTRDEREF);
    dmat result = {n_rows,
                   n_cols,
                   !own_data,
                   n_rows,
                   data};
    return result;
}
/**
 * Create a cmat from foreign data. Assumes the colstride of the data is
 * n_rows.
 *
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 * @param data Pointer to data storage
 *
 * @return dmat
 */
static inline cmat cmat_foreign_data(us n_rows,
                                     us n_cols,
                                     c* data,
                                     bool own_data) {

    dbgassert(data,NULLPTRDEREF);
    cmat result = {n_rows,
                   n_cols,
                   !own_data,
                   n_rows,
                   data};
    return result;
}

/**
 * Creates a cmat from foreign data. Does not copy the data, but only
 * initializes the row pointers. Assumes column-major ordering for the
 * data. Please do not keep this one alive after the data has been
 * destroyed. Assumes the column colstride equals to n_rows.
 *
 * @param n_rows
 * @param n_cols
 * @param data
 *
 * @return
 */
static inline cmat cmat_foreign(cmat* other) {
    dbgassert(other,NULLPTRDEREF);
    cmat result = {other->n_rows,
                   other->n_cols,
                   true,
                   other->colstride,
                   other->_data};
    return result;
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
#define vd_free dmat_free

/**
 * Free's data of dmat. Safe to run on sub-matrices as well.
 *
 * @param m Matrix to free
 */
static inline void cmat_free(cmat* m) {
    dbgassert(m,NULLPTRDEREF);
    if(!(m->_foreign_data)) a_free(m->_data);
}
#define vc_free cmat_free

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
    dbgassert(to && from,NULLPTRDEREF);
    dbgassert(to->n_cols == from->n_cols,SIZEINEQUAL);
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
                    parent->n_rows, // This is the colstride to get to
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
                    parent->n_rows, // This is the colstride to get to
                    // the next column.
                    getcmatval(parent,startrow,startcol)};

    return result;
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
 * Allocate a new array, with size based on other. 
 *
 * @param[in] from: Array to copy
 */
static inline dmat dmat_alloc_from_dmat(const dmat* from) {
    assertvalidptr(from);
    dmat thecopy = dmat_alloc(from->n_rows, from->n_cols);
    return thecopy;
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
 * Copy contents of one vector to another
 *
 * @param to : Vector to write to
 * @param from : Vector to read from
 */
static inline void vd_copy(vd* to,const vd* from) {
    dbgassert(to && from,NULLPTRDEREF);
    assert_vx(to);
    assert_vx(from);
    dmat_copy(to,from);
}
/**
 * Copy contents of one vector to another
 *
 * @param to : Vector to write to
 * @param from : Vector to read from
 */
static inline void vc_copy(vc* to,const vc* from) {
    dbgassert(to && from,NULLPTRDEREF);
    assert_vx(to);
    assert_vx(from);
    cmat_copy(to,from);
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
    vd res = {x->n_rows,1,true,x->colstride,getdmatval(x,0,col)};
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
    vc res = {x->n_rows,1,true,x->colstride,getcmatval(x,0,col)};
    return res;
}

/**
 * Compute the complex conjugate of b and store result in a
 *
 * @param a
 * @param b
 */
static inline void cmat_conj(cmat* a,const cmat* b) {
    fsTRACE(15);
    dbgassert(a && b,NULLPTRDEREF);
    dbgassert(a->n_cols == b->n_cols,SIZEINEQUAL);
    dbgassert(a->n_rows == b->n_rows,SIZEINEQUAL);
    for(us col=0;col<a->n_cols;col++) {
        carray_conj(getcmatval(a,0,col),getcmatval(b,0,col),a->n_rows);
    }
    feTRACE(15);
}

/**
 * Take the complex conjugate of x, in place
 *
 * @param x
 */
static inline void cmat_conj_inplace(cmat* x) {
    dbgassert(x,NULLPTRDEREF);
    for(us col=0;col<x->n_cols;col++) {
        c_conj_inplace(getcmatval(x,0,col),x->n_rows);
    }
}

/**
 * Computes the maximum value for each row, returns a vector with maximum
 * values for each column in the matrix.
 *
 * @param x
 */
static inline vd dmat_max(const dmat x) {
    vd max_vals = vd_alloc(x.n_cols);
    d max_val = -d_inf;
    for(us j=0; j< x.n_cols; j++) {
        for(us i=0; i< x.n_rows; i++) {
            max_val = *getdmatval(&x, i, j) < max_val? max_val : *getdmatval(&x, i,j);
        }
        *getvdval(&max_vals, j) = max_val;
    }
    return max_vals;
}


#ifdef LASP_DEBUG
void print_cmat(const cmat* m);
void print_dmat(const dmat* m);
#define print_vc(x) assert_vx(x) print_cmat(x)
#define print_vd(x) assert_vx(x) print_dmat(x)

#else
#define print_cmat(m)
#define print_dmat(m)
#define print_vc(m)
#define print_vd(m)
#endif

#endif // LASP_MAT_H
//////////////////////////////////////////////////////////////////////
