// lasp_math_raw.c
//
// last-edit-by: J.A. de Jong 
// 
// Description:
// Operations working on raw arrays of floating point numbers
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-5)
#include "lasp_math_raw.h"
#if LASP_USE_BLAS
#include <cblas.h>
#endif

void d_elem_prod_d(d res[],
                   const d arr1[],
                   const d arr2[],
                   const us size) {

    #if LASP_USE_BLAS == 1

    #if LASP_DEBUG

    if(arr1 == arr2) {
        DBGWARN("d_elem_prod_d: Array 1 and array 2 point to the same"
                " memory. This results in pointer aliasing, for which"
                " testing is still to be done. Results might be"
                " unrealiable.");
    }

    #endif


    #if LASP_DOUBLE_PRECISION
    #define elem_prod_fun cblas_dsbmv
    #else
    #define elem_prod_fun cblas_ssbmv
    #endif
    /* These parameters do not matter for this specific case */
    const CBLAS_ORDER  mat_order= CblasColMajor;
    const CBLAS_UPLO   uplo = CblasLower;

    /* Extra multiplication factor */
    const d alpha = 1.0;

    /* void cblas_dsbmv(OPENBLAS_CONST enum CBLAS_ORDER order, */
    /*                  OPENBLAS_CONST enum CBLAS_UPLO Uplo, */
    /*                  OPENBLAS_CONST blasint N, */
    /*                  OPENBLAS_CONST blasint K, */
    /*                  OPENBLAS_CONST double alpha, */
    /*                  OPENBLAS_CONST double *A, */
    /*                  OPENBLAS_CONST blasint lda, */
    /*                  OPENBLAS_CONST double *X, */
    /*                  OPENBLAS_CONST blasint incX, */
    /*                  OPENBLAS_CONST double beta, */
    /*                  double *Y, */
    /*                  OPENBLAS_CONST blasint incY); */

    elem_prod_fun(mat_order,
                  uplo,
                  (blasint) size,
                  0,             // Just the diagonal; 0 super-diagonal bands
                  alpha,        /* Multiplication factor alpha */
                  arr1,
                  1,            /* LDA */
                  arr2,         /* x */
                  1, /* incX = 1 */
                  0.0,          /* Beta */
                  res,    /* The Y matrix to write to */
                  1); /* incY */
    #undef elem_prod_fun

    #else  /* No blas routines, routine is very simple, but here we
            * go! */
    DBGWARN("Performing slow non-blas vector-vector multiplication");
    for(us i=0;i<size;i++) {
        res[i] = arr1[i]*arr2[i];
    }
    #endif
}

void c_hadamard(c res[],
                const c arr1[],
                const c arr2[],
                const us size) {

    fsTRACE(15);
    uVARTRACE(15,size);
    dbgassert(arr1 && arr2 && res,NULLPTRDEREF);
    
    #if LASP_USE_BLAS == 1

    #if LASP_DEBUG

    if(arr1 == arr2) {
        DBGWARN("c_elem_prod_c: Array 1 and array 2 point to the same"
                " memory. This results in pointer aliasing, for which"
                " testing is still to be done. Results might be"
                " unrealiable.");
    }

    #endif  /* LASP_DEBUG */


    #if LASP_DOUBLE_PRECISION
    #define elem_prod_fun cblas_zgbmv
    #else
    #define elem_prod_fun cblas_cgbmv
    #endif

    c alpha = 1.0;
    c beta = 0.0;

    TRACE(15,"Calling " annestr(elem_prod_fun));
    uVARTRACE(15,size);
    elem_prod_fun(CblasColMajor,
                  CblasNoTrans,
                  (blasint) size, /* M: Number of rows */
                  (blasint) size, /* B: Number of columns */
                  0,              /* KL: Number of sub-diagonals */
                  0,              /* KU: Number of super-diagonals */
                  (d*) &alpha,    /* Multiplication factor */
                  (d*) arr2,      /* A */
                  1,              /* LDA */
                  (d*) arr1,      /* x */
                  1,              /* incX = 1 */
                  (d*) &beta,
                  (d*) res,    /* The Y matrix to write to */
                  1);          /* incY (increment in res) */

    #undef elem_prod_fun

    #else  /* No blas routines, routine is very simple, but here we
            * go! */
    DBGWARN("Performing slower non-blas vector-vector multiplication");
    for(us i=0;i<size;i++) {
        res[i] = arr1[i]*arr2[i];
    }
    #endif
    feTRACE(15);
}




//////////////////////////////////////////////////////////////////////

