// lasp_alg.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// (Linear) algebra routines on matrices and vectors
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_ALG_H
#define LASP_ALG_H
#include "lasp_math.h"

/** 
 * Compute the dot product of two vectors of floats
 *
 * @param a First vector
 * @param b Second second vector
 * @return dot product as float
 */
static inline d vd_dot(const vd * a,const vd* b) {
    dbgassert(a && b,NULLPTRDEREF);
    dbgassert(a->size == b->size,SIZEINEQUAL);
    return d_dot(getvdval(a,0),getvdval(b,0),a->size);
}

/** 
 * y = fac * y
 *
 * @param y 
 * @param fac scale factor
 */
static inline void dmat_scale(dmat* y,const c fac){
    dbgassert(y,NULLPTRDEREF);
    for(us col=0;col<y->n_cols;col++) {
        d_scale(getdmatval(y,0,col),fac,y->n_rows);
    }

}
/** 
 * y = fac * y
 *
 * @param y 
 * @param fac scale factor
 */
static inline void cmat_scale(cmat* y,const c fac){
    dbgassert(y,NULLPTRDEREF);
    dbgassert(y,NULLPTRDEREF);
    for(us col=0;col<y->n_cols;col++) {
        c_scale(getcmatval(y,0,col),fac,y->n_rows);
    }
}

/** 
 * x = x + fac*y
 *
 * @param x 
 * @param y 
 * @param fac 
 */
static inline void dmat_add_dmat(dmat* x,dmat* y,d fac) {
    dbgassert(x && y,NULLPTRDEREF);
    dbgassert(x->n_cols == y->n_cols,SIZEINEQUAL);
    dbgassert(x->n_rows == y->n_rows,SIZEINEQUAL);
    for(us col=0;col<y->n_cols;col++) {
        d_add_to(getdmatval(x,0,col),
                 getdmatval(y,0,col),fac,x->n_rows);
    }
}
/** 
 * x = x + fac*y
 *
 * @param[in,out] x 
 * @param[in] y 
 * @param[in] fac 
 */
static inline void cmat_add_cmat(cmat* x,cmat* y,c fac) {
    dbgassert(x && y,NULLPTRDEREF);
    dbgassert(x->n_cols == y->n_cols,SIZEINEQUAL);
    dbgassert(x->n_rows == y->n_rows,SIZEINEQUAL);
    for(us col=0;col<y->n_cols;col++) {
        c_add_to(getcmatval(x,0,col),
                 getcmatval(y,0,col),fac,x->n_rows);
    }
}

/** 
 * Compute the element-wise (Hadamard) product of a and b, and store
 * in result
 *
 * @param[out] result 
 * @param[in] a 
 * @param[in] b 
 */
static inline void vd_elem_prod(vd* result,const vd* a,const vd* b) {
    dbgassert(result  && a && b,NULLPTRDEREF);
    dbgassert(result->size==a->size,SIZEINEQUAL);
    dbgassert(b->size==a->size,SIZEINEQUAL);
    d_elem_prod_d(getvdval(result,0),
                  getvdval(a,0),
                  getvdval(b,0),a->size);
}
/** 
 * Compute the element-wise (Hadamard) product of a and b, and store
 * in result
 *
 * @param[out] result 
 * @param[in] a 
 * @param[in] b 
 */
static inline void vc_hadamard(vc* result,const vc* a,const vc* b) {
    fsTRACE(15);
    dbgassert(result  && a && b,NULLPTRDEREF);
    dbgassert(result->size==a->size,SIZEINEQUAL);
    dbgassert(b->size==a->size,SIZEINEQUAL);
    c_hadamard(getvcval(result,0),
               getvcval(a,0),
               getvcval(b,0),
               a->size);
    
    check_overflow_vx(*result);
    check_overflow_vx(*a);
    check_overflow_vx(*b);    
    feTRACE(15);
}

/** 
 * Compute the element-wise (Hadamard) product of a and b, and store
 * in result
 *
 * @param[out] result 
 * @param[in] a 
 * @param[in] b 
 */
static inline void cmat_hadamard(cmat* result,
                                 const cmat* a,
                                 const cmat* b) {
    fsTRACE(15);
    dbgassert(result  && a && b,NULLPTRDEREF);

    dbgassert(result->n_rows==a->n_rows,SIZEINEQUAL);
    dbgassert(result->n_cols==a->n_cols,SIZEINEQUAL);
    dbgassert(b->n_rows==a->n_rows,SIZEINEQUAL);
    dbgassert(b->n_cols==a->n_cols,SIZEINEQUAL);

    for(us col=0;col<result->n_cols;col++) {
        c_hadamard(getcmatval(result,0,col),
                   getcmatval(a,0,col),
                   getcmatval(b,0,col),
                   a->n_rows);
    }

    feTRACE(15);
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

/** 
 * Compute the norm of the difference of two complex matrices
 *
 * @param A 
 * @param B 
 */
d cmat_normdiff(const cmat* A,const cmat* B);

/** 
 * Computes the Kronecker product of a kron b, stores result in result.
 *
 * @param a a
 * @param b b
 * @param result a kron b
 */
void kronecker_product(const cmat* a,const cmat* b,cmat* result);

#endif // LASP_ALG_H
//////////////////////////////////////////////////////////////////////
