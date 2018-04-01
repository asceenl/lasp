// lasp_alg.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// (Linear) algebra routine implementations
//////////////////////////////////////////////////////////////////////
#include "lasp_alg.h"

void cmv_dot(const cmat* A,const vc* restrict x,vc* restrict b){
    dbgassert(A && x && b,NULLPTRDEREF);
    assert_vx(b);
    assert_vx(x);
    dbgassert(A->n_cols == x->n_rows,SIZEINEQUAL);
    dbgassert(A->n_rows == b->n_rows,SIZEINEQUAL);

    #if LASP_USE_BLAS == 1
    dbgassert(false,"Untested function. Is not functional for strides");
    /* typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER; */
    /* typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE; */
    /* 
       void cblas_zgemv(OPENBLAS_CONST enum CBLAS_ORDER order,
       OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,
       OPENBLAS_CONST blasint m,
       OPENBLAS_CONST blasint n,
       OPENBLAS_CONST double *alpha,
       OPENBLAS_CONST double  *a,
       OPENBLAS_CONST blasint lda,
       OPENBLAS_CONST double  *x,
       OPENBLAS_CONST blasint incx,
       OPENBLAS_CONST double *beta,
       double  *y,
       OPENBLAS_CONST blasint incy);
    */
    c alpha = 1.0;
    c beta = 0.0;
    cblas_zgemv(CblasColMajor,
                CblasNoTrans,
                A->n_rows,
                A->n_cols,
                (d*) &alpha,			/* alpha */
                (d*) A->_data,		/* A */
                A->n_rows,		/* lda */
                (d*) x->_data,		/*  */
                1,
                (d*) &beta,			/* beta */
                (d*) b->_data,
                1);
				
				
				
    #else
    size_t i,j;

    vc_set(b,0.0);

    iVARTRACE(20,A->n_cols);
    iVARTRACE(20,A->n_rows);

    for(j=0;j<A->n_cols;j++){
        for(i=0;i<A->n_rows;i++) {

            c* Aij = getcmatval(A,i,j);
            b->_data[i] += *Aij * *getvcval(x,j);

        }

    }


    #endif
}

/// The code below here is not yet worked on. Should be improved to
/// directly couple to openblas, instead of using lapacke.h
#if 0

/* These functions can be directly linked to openBLAS */
#define lapack_complex_double   double _Complex
#define lapack_complex_float   float _Complex

#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102

#define LAPACK_WORK_MEMORY_ERROR       -1010
#define LAPACK_TRANSPOSE_MEMORY_ERROR  -1011

typedef int lapack_int;

int LAPACKE_cgelss( int matrix_layout, int m, int n,
                    int nrhs, lapack_complex_float* a,
                    int lda, lapack_complex_float* b,
                    int ldb, float* s, float rcond,
                    int* rank );
int LAPACKE_zgelss( int matrix_layout, int m, int n,
                    int nrhs, lapack_complex_double* a,
                    int lda, lapack_complex_double* b,
                    int ldb, double* s, double rcond,
                    int* rank );

lapack_int LAPACKE_zgels( int matrix_layout, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_complex_double* b, lapack_int ldb );




#if LASP_FLOAT == 64

#define lapack_gelss LAPACKE_zgelss
#define lapack_gels LAPACKE_zgels
#else

#define lapack_gelss LAPACKE_cgelss
#endif

#define max(a,b) ((a)>(b)?(a):(b))


/* int lsq_solve(const cmat* A,const vc* b,vc* x){ */
    
/*     int rv; */
/*     /\* M: number of rows of matrix *\/ */
/*     /\* N: Number of columns *\/ */
/*     /\* Norm: L2|b-A*x| *\/ */
/*     /\* NRHS: Number of right hand sides: Number of columns of matrix B *\/ */

/*     assert(A->n_rows>=A->n_cols); */
/*     assert(x->size == A->n_cols); */
/*     assert(b->size == A->n_rows); */
	
/*     int info; */
	
/*     size_t lda = max(1,A->n_rows); */
/*     size_t ldb = max(lda,A->n_cols); */
	
/*     /\* Make explicit copy of matrix A data, as it will be overwritten */
/*      * by lapack_gels *\/ */
/*     c* A_data = Pool_allocatec(&lsq_solve_pool,A->n_rows*A->n_cols); */
/*     c_copy(A_data,A->data,A->n_cols*A->n_rows); */

/*     c* work_data = Pool_allocatec(&lsq_solve_pool,b->size); */
/*     c_copy(work_data,b->data,b->size); */
	
/*     /\* Lapack documentation says: *\/ */
/*     /\* 	if TRANS = 'N' and m >= n, rows 1 to n of B contain the least */
/*         squares solution vectors; the residual sum of squares for the */
/*         solution in each column is given by the sum of squares of the */
/*         modulus of elements N+1 to M in that column; */
/*     *\/ */

	
/*     /\* We always assume one RHS column *\/ */
/*     const int nrhs = 1; */

/*     /\* General Least Squares Solve *\/ */
/*     info = lapack_gels(LAPACK_COL_MAJOR, /\* Column-major ordering *\/ */
/*                        'N', */
/*                        A->n_rows,	/\* Number of rows in matrix *\/ */
/*                        A->n_cols,	/\* Number of columns *\/ */
/*                        nrhs,   /\* nrhs, which is number_mics *\/ */
/*                        A_data, /\* The A-matrix *\/ */
/*                        lda,	  /\* lda: the leading dimension of matrix A *\/ */
/*                        work_data,	  /\* The b-matrix *\/ */
/*                        ldb);  /\* ldb: the leading dimension of b: max(1,M,N) *\/ */
		
/*     if(info==0){ */
/*         c_copy(x->data,work_data,x->size); */
/*         rv = SUCCESS; */
/*     } */
/*     else { */
/*         memset(x->data,0,x->size); */
/*         WARN("LAPACK INFO VALUE"); */
/*         printf("%i\n", info ); */
/*         TRACE(15,"Solving least squares problem failed\n"); */

/*         rv = FAILURE; */
/*     } */

/*     return rv; */
    
/* } */

/* d c_normdiff(const cmat* A,const cmat* B) { */

/*     TRACE(15,"c_normdif"); */
	
/*     dbgassert(A->n_cols==B->n_cols,"Number of columns of A and B " */
/*               "should be equal"); */
/*     dbgassert(A->n_rows==B->n_rows,"Number of rows of A and B " */
/*               "should be equal"); */

/*     size_t size = A->n_cols*A->n_rows; */

/*     vc diff_temp = vc_al[MAX_MATRIX_SIZE]; */

/*     c_copy(diff_temp,A->data,size); */

/*     c alpha = -1.0; */

/*     /\* This routine computes y <- alpha*x + beta*y *\/ */
	
	
/*     /\* void cblas_zaxpy(OPENBLAS_CONST blasint n,  *\/ */
/*     /\* 				 OPENBLAS_CONST double *alpha, *\/ */
/*     /\* 				 OPENBLAS_CONST double *x, *\/ */
/*     /\* 				 OPENBLAS_CONST blasint incx, *\/ */
/*     /\* 				 double *y, *\/ */
/*     /\* 				 OPENBLAS_CONST blasint incy); *\/ */

/*     cblas_zaxpy(size, */
/*                 (d*) &alpha, */
/*                 (d*) B->data, */
/*                 1, */
/*                 (d*) diff_temp, */
/*                 1 ); */
	
/*     return c_norm(diff_temp,size); */
/* } */
#endif  /* if 0 */

//////////////////////////////////////////////////////////////////////
