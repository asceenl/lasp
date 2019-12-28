// lasp_siggen.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// Signal generator implementation
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-5)
#include "lasp_alloc.h"
#include "lasp_assert.h"
#include "lasp_mat.h"

#define PRIVATE_SIZE 32

typedef enum {
    SINEWAVE = 0,
    WHITENOISE,
    PINKNOISE,
    SWEEP,

} SignalType;


typedef struct {
    SignalType signaltype;
    d fs; // Sampling frequency [Hz]
    void* private;

    char private_data[PRIVATE_SIZE];

} Siggen;

typedef struct { 
    d curtime;
    d omg;
} SinewaveSpecific;


Siggen* Siggen_Sinewave_create(const d fs, const d freq) {
    fsTRACE(15);

    Siggen* sinesiggen = a_malloc(sizeof(Siggen));
    sinesiggen->signaltype = SINEWAVE;
    sinesiggen->fs = fs;
    sinesiggen->private = sinesiggen->private_data;
    SinewaveSpecific* sp = (SinewaveSpecific*) sinesiggen->private;
    sp->curtime = 0;
    sp->omg = 2*number_pi*freq;

    
    feTRACE(15);
    return sinesiggen;
}
Siggen* Siggen_Whitenoise_create() {
    fsTRACE(15);

    Siggen* whitenoise = a_malloc(sizeof(Siggen));
    whitenoise->signaltype = WHITENOISE;
    
    feTRACE(15);
    return whitenoise;
}
void Siggen_free(Siggen* siggen) {
    fsTRACE(15);
    assertvalidptr(siggen);

    if(siggen->signaltype == SWEEP) {
        /* Sweep specific stuff here */
    }
    
    a_free(siggen);
    feTRACE(15);
}
static void Sinewave_genSignal(Siggen* siggen, SinewaveSpecific* sine, vd* samples) {
    fsTRACE(15);
    assertvalidptr(sine);
    d ts = 1/siggen->fs;
    d omg = sine->omg;

    d curtime = sine->curtime;
    for(us i =0; i< samples->n_rows; i++) {
        setvecval(samples, i, sin(omg*curtime));
        curtime = curtime + ts;
    }
    sine->curtime = curtime;
    feTRACE(15);
}

static d gaussrand() {
	static d V1, V2, S;
	static int phase = 0;
	d X;

	if(phase == 0) {
		do {
			d U1 = (d)rand() / RAND_MAX;
			d U2 = (d)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
			} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return X;
}
static void Whitenoise_genSignal(Siggen* siggen, vd* samples) {
    for(us i =0; i< samples->n_rows; i++) {
        d rn = gaussrand();
        setvecval(samples, i, rn);
    }

}
void Siggen_genSignal(Siggen* siggen,vd* samples) {
    fsTRACE(15);
    assertvalidptr(siggen);
    assert_vx(samples);
    d fs = siggen->fs;

    switch(siggen->signaltype) {
        case SINEWAVE:
            Sinewave_genSignal(siggen, (SinewaveSpecific*) siggen->private, 
                    samples);

            break;
        case WHITENOISE:
            Whitenoise_genSignal(siggen, samples);

    }
     
    feTRACE(15);
}



//////////////////////////////////////////////////////////////////////
