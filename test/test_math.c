// test_bf.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////

#include "lasp_fft.h"
#include "lasp_mat.h"
#include "lasp_tracer.h"
#include "lasp_alg.h"

int main() {

    iVARTRACE(15,getTracerLevel());

    c a[5];
    c_set(a,1-I,5);

    /* print_vc(&a); */

    vc b = vc_alloc(5);    
    vc_set(&b,2);


    printf("b:\n");    
    print_vc(&b);    


    vc c1 = vc_alloc(5);
    /* vc_set(&c1,10); */
    /* c_add_to(c1.ptr,a.ptr,1,3); */
    c_hadamard(c1._data,a,b._data,5);



    printf("c1:\n");
    print_vc(&c1); 
    
    vc_free(&b);


    
    return 0;
}

//////////////////////////////////////////////////////////////////////


