// test_bf.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////

#include "fft.h"
#include "ascee_math.h"
#include "ascee_tracer.h"

int main() {

    iVARTRACE(15,getTracerLevel());

    vc a = vc_alloc(5);
    vc_set(&a,2+3*I);

    c_conj_inplace(a.data,a.size);
    print_vc(&a);

    vc b = vc_alloc(5);    

    c_conj_c(b.data,a.data,5);

    print_vc(&b);    

    vc_free(&a);
    vc_free(&b);    
    return 0;
}

//////////////////////////////////////////////////////////////////////


