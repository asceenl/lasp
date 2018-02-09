// ascee_math.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// 
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-10)
#include "ascee_math.h"

#include "ascee_assert.h"
#include "ascee_math.h"
#include "ascee_tracer.h"

#include <math.h>

#ifdef ASCEE_DEBUG
void print_dmat(const dmat* m) {
    feTRACE(50);
    size_t row,col;
    for(row=0;row<m->n_rows;row++){
        indent_trace();
        for(col=0;col<m->n_cols;col++){
            d val = *getdmatval(m,row,col);
            printf("%c%2.2e ", val<0?'-':' ' ,d_abs(val));
			
        }
        printf("\n");

    }
    fsTRACE(50);
}
void print_cmat(const cmat* m) {
    feTRACE(50);
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
void print_vc(const vc* m) {
    fsTRACE(50);
    size_t row;

    for(row=0;row<m->size;row++){
        c val = *getvcval(m,row);

        d rval = creal(val);
        d ival = cimag(val);
        indent_trace();
        printf("%c%2.2e%c%2.2ei ",rval< 0 ?'-': ' ', d_abs(rval),ival<0 ? '-' : '+',d_abs(ival) ) ;
        printf("\n");

    }
    feTRACE(50);    
}
void print_vd(const vd* m) {
    fsTRACE(50);
    size_t row;
    iVARTRACE(20,m->size);
    for(row=0;row<m->size;row++){
	
        d rval = m->ptr[row];
        indent_trace();
        printf("%c%2.2e ",rval< 0 ? '\r': ' ',rval);
        printf("\n");
    }
    feTRACE(50);
}

#endif

//////////////////////////////////////////////////////////////////////
