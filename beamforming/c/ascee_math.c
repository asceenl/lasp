// si_math.c
//
// last-edit-by: J.A. de Jong 
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-10)

#include "ascee_assert.h"
#include "ascee_math.h"
#include "tracer.h"

#if ASCEE_USE_BLAS
#include <cblas.h>
#endif

#include <math.h>

#ifdef ASCEE_DEBUG
void print_cmat(const cmat* m) {
    size_t row,col;
    for(row=0;row<m->n_rows;row++){
        for(col=0;col<m->n_cols;col++){
            c val = m->data[row+m->n_rows*col];

            d rval = creal(val);
            d ival = cimag(val);
			
            printf("%c%2.2e%c%2.2ei ",rval< 0 ?'-': ' ', d_abs(rval),ival<0 ? '-' : '+',d_abs(ival) ) ;
			
        }
        printf("\n");

    }
}
void print_vc(const vc* m) {
    TRACE(20,"print_vc");
    size_t row;

    for(row=0;row<m->size;row++){
	
        d rval = creal(m->data[row]);
        d ival = cimag(m->data[row]);

        printf("%c%2.2e%c%2.2ei ",rval< 0 ?'-': ' ', d_abs(rval),ival<0 ? '-' : '+',d_abs(ival) ) ;
        printf("\n");

    }
}
void print_vd(const vd* m) {
    TRACE(20,"print_vd");
    size_t row;
    iVARTRACE(20,m->size);
    for(row=0;row<m->size;row++){
	
        d rval = m->data[row];

        printf("%c%2.2e ",rval< 0 ? '\r': ' ',rval);
        printf("\n");
    }
}
void print_dmat(const dmat* m) {
    size_t row,col;
    for(row=0;row<m->n_rows;row++){
        for(col=0;col<m->n_cols;col++){
            d val = m->data[row+m->n_rows*col];
            printf("%c%2.2e ", val<0?'-':' ' ,d_abs(val));
			
        }
        printf("\n");

    }
}
#endif

void d_elem_prod_d(d res[],
                   const d arr1[],
                   const d arr2[],
                   const us size) {

    #if ASCEE_USE_BLAS

    #if ASCEE_DEBUG

    if(arr1 == arr2) {
        DBGWARN("d_elem_prod_d: Array 1 and array 2 point to the same"
                " memory. This results in pointer aliasing, for which"
                " testing is still to be done. Results might be"
                " unrealiable.");
    }

    #endif


    #if ASCEE_DOUBLE_PRECISION
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

void c_elem_prod_c(c res[],
                   const c arr1[],
                   const c arr2[],
                   const us size) {

    TRACE(15,"c_elem_prod_c");
    uVARTRACE(15,size);
    
    #if ASCEE_USE_BLAS

    #if ASCEE_DEBUG

    if(arr1 == arr2) {
        DBGWARN("c_elem_prod_c: Array 1 and array 2 point to the same"
                " memory. This results in pointer aliasing, for which"
                " testing is still to be done. Results might be"
                " unrealiable.");
    }

    #endif  /* ASCEE_DEBUG */


    #if ASCEE_DOUBLE_PRECISION
    #define elem_prod_fun cblas_zgbmv
    #else
    #define elem_prod_fun cblas_cgbmv
    #endif

    /* These parameters do not matter for this specific case */
    const CBLAS_ORDER  mat_order= CblasColMajor;
    const CBLAS_TRANSPOSE tr = CblasNoTrans;

    const c alpha = 1.0;
    const c beta = 0.0;
    TRACE(15,"Calling " annestr(elem_prod_fun));
    
    elem_prod_fun(mat_order,
                  tr,
                  (blasint) size, /* M: Number of rows */
                  (blasint) size, /* B: Number of columns */
                  0,              /* KL: Number of sub-diagonals */
                  0,              /* KU: Number of super-diagonals */
                  (d*) &alpha,        /* Multiplication factor */
                  (d*) arr2,          /* A */
                  1,            /* LDA */
                  (d*) arr1,    /* x */
                  1, /* incX = 1 */
                  (d*) &beta,
                  (d*) res,    /* The Y matrix to write to */
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


void cmv_dot(const cmat* A,const vc* restrict x,vc* restrict b){

    assert(A->n_rows == b->size);
    assert(A->n_cols == x->size);
	
    #if ASCEE_USE_BLAS == 1

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
                (d*) A->data,		/* A */
                A->n_rows,		/* lda */
                (d*) x->data,		/*  */
                1,
                (d*) &beta,			/* beta */
                (d*) b->data,
                1);
				
				
				
    #else
    size_t i,j;
    size_t n_rows = A->n_rows;

    vc_set(b,0.0);

    iVARTRACE(20,A->n_cols);
    iVARTRACE(20,A->n_rows);

    for(j=0;j<A->n_cols;j++){
        for(i=0;i<A->n_rows;i++) {

            c* Aij = &A->data[i+j*n_rows];
            b->data[i] += *Aij * x->data[j];

        }

    }


    #endif
}
	
void kronecker_product(const cmat* a,const cmat* b,cmat* result){

    assert(result->n_rows == a->n_rows*b->n_rows);
    assert(result->n_cols == a->n_cols*b->n_cols);

    c a_rs;
    c b_vw;

    int r_col;
    int r_row;

    for(size_t r=0; r< a->n_rows;r++){

        for(size_t s=0; s <a->n_cols;s++) {

            for(size_t v=0;v < b->n_rows; v++) {

                for(size_t w=0;w < b->n_cols;w++) {
					
                    a_rs = *getcmatval(a,r,s);
                    b_vw = *getcmatval(b,v,w);

                    r_row = b->n_rows*r+v;
                    r_col = b->n_cols*s+w;
					
                    result->data[r_row + r_col * result->n_rows] = a_rs * b_vw;

                }
            }
        }
    }
} /* void kronecker_product */

/* #include <lapacke.h> */
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




#if ASCEE_FLOAT == 64

#define lapack_gelss LAPACKE_zgelss
#define lapack_gels LAPACKE_zgels
#else

#define lapack_gelss LAPACKE_cgelss
#endif

#define max(a,b) ((a)>(b)?(a):(b))


/* int lsq_solve(const cmat* A,const vc* b,vc* x){ */
    
/*     POOL_INIT(lsq_solve_pool); */
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

/*     Pool_free(&lsq_solve_pool,A_data); */
/*     Pool_free(&lsq_solve_pool,work_data); */
/*     POOL_EXIT(lsq_solve_pool,15);     */
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

//////////////////////////////////////////////////////////////////////

