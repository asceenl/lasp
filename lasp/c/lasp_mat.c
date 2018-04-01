// lasp_mat.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// 
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-10)
#include "lasp_mat.h"
#include "lasp_assert.h"
#include "lasp_tracer.h"

#include <math.h>

#ifdef LASP_DEBUG
void print_dmat(const dmat* m) {
    fsTRACE(50);
    size_t row,col;
    for(row=0;row<m->n_rows;row++){
        indent_trace();
        for(col=0;col<m->n_cols;col++){
            d val = *getdmatval(m,row,col);
            printf("%c%2.2e ", val<0?'-':' ' ,d_abs(val));
			
        }
        printf("\n");

    }
    feTRACE(50);
}
void print_cmat(const cmat* m) {
    fsTRACE(50);
    size_t row,col;
    for(row=0;row<m->n_rows;row++){
        indent_trace();
        for(col=0;col<m->n_cols;col++){
            c val = *getcmatval(m,row,col);

            d rval = creal(val);
            d ival = cimag(val);
			
            printf("%c%2.2e%c%2.2ei ",rval< 0 ?'-': ' ',
                   d_abs(rval),ival<0 ? '-' : '+',d_abs(ival) ) ;
			
        }
        printf("\n");

    }
    feTRACE(50);
}

#endif

//////////////////////////////////////////////////////////////////////
