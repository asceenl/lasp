// test_bf.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////

#include "lasp_fft.h"
#include "lasp_tracer.h"


int main() {

    setTracerLevel(0);

    iVARTRACE(15,getTracerLevel());

    Fft* fft = Fft_create(100000);

    /* Fft_fft(fft,NULL,NULL); */

    Fft_free(fft);

    return 0;
}




//////////////////////////////////////////////////////////////////////


