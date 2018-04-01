// test_bf.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#include "lasp_mat.h"

int main() {

    iVARTRACE(15,getTracerLevel());
    /* vd vec1 = vd_alloc(3); */
    /* vd_set(&vec1,2); */

    /* vd vec2 = vd_alloc(3); */
    /* vd_set(&vec2,3); */

    /* print_vd(&vec1); */

    /* vd res = vd_alloc(3); */
    /* d_elem_prod_d(res.data,vec1.data,vec2.data,3); */

    /* print_vd(&res); */


    vc vc1 = vc_alloc(3);
    vc_set(&vc1,2+2I);
    print_vc(&vc1);

    vc vc2 = vc_alloc(3);
    vc_set(&vc2,2-2I);
    setvecval(&vc2,0,10);
    print_vc(&vc2);


    vc res2 = vc_alloc(3);
    c_hadamard(res2._data,vc1._data,vc2._data,3);

    print_vc(&res2);



}

//////////////////////////////////////////////////////////////////////
