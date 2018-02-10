// test_bf.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////

#include "fft.h"
#include "ascee_tracer.h"


int main() {

    setTracerLevel(0);

    iVARTRACE(15,getTracerLevel());

    Fft* fft = Fft_alloc(100000);

    /* Fft_fft(fft,NULL,NULL); */

    Fft_free(fft);

    return 0;
}




//////////////////////////////////////////////////////////////////////


