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

#define PRIVATE_SIZE 64

typedef enum {
    SINEWAVE = 0,
    WHITENOISE,
    SWEEP,

} SignalType;

typedef struct {
    SignalType signaltype;
    d fs; // Sampling frequency [Hz]
    d level_amp;
    void* private;

    char private_data[PRIVATE_SIZE];

} Siggen;

typedef struct { 
    d curtime;
    d omg;
} SinewaveSpecific;

typedef struct {
    d V1, V2, S;
    int phase;
} WhitenoiseSpecific;

static d level_amp(d level_dB){
    return pow(10, level_dB/20);
}

Siggen* Siggen_create(SignalType type, const d fs,const d level_dB) {

    fsTRACE(15);

    Siggen* siggen = a_malloc(sizeof(Siggen));
    siggen->signaltype = type;
    siggen->fs = fs;
    siggen->private = NULL;
    siggen->level_amp = level_amp(level_dB);

    feTRACE(15);
    return siggen;
}

Siggen* Siggen_Sinewave_create(const d fs, const d freq,const d level_dB) {
    fsTRACE(15);

    Siggen* sinesiggen = Siggen_create(SINEWAVE, fs, level_dB);
    sinesiggen->private = sinesiggen->private_data;
    SinewaveSpecific* sp = (SinewaveSpecific*) sinesiggen->private;
    sp->curtime = 0;
    sp->omg = 2*number_pi*freq;

    feTRACE(15);
    return sinesiggen;
}

Siggen* Siggen_Whitenoise_create(const d fs, const d level_dB) {
    fsTRACE(15);

    Siggen* whitenoise = Siggen_create(WHITENOISE, fs, level_dB);
    whitenoise->private = whitenoise->private_data;
    dbgassert(sizeof(WhitenoiseSpecific) <= sizeof(whitenoise->private_data), "Allocated memory too small");
    WhitenoiseSpecific* wn = whitenoise->private;
    wn->phase = 0;
    wn->V1 = 0;
    wn->V2 = 0;
    wn->S = 0;

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
        setvecval(samples, i, siggen->level_amp*sin(omg*curtime));
        curtime = curtime + ts;
    }
    sine->curtime = curtime;
    feTRACE(15);
}

static void Whitenoise_genSignal(Siggen* siggen, WhitenoiseSpecific* wn, vd* samples) {
    fsTRACE(15);
    d X;
    d S = wn->S;
    d V1 = wn->V1;
    d V2 = wn->V2;
    int phase = wn->phase;

    for(us i =0; i< samples->n_rows; i++) {

        if(wn->phase == 0) {
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

        setvecval(samples, i, siggen->level_amp*X);
    }
    wn->S = S;
    wn->V1 = V1;
    wn->V2 = V2;
    wn->phase = phase;
    feTRACE(15);

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
            Whitenoise_genSignal(siggen, (WhitenoiseSpecific*) siggen->private,
                    samples);
            break;
        case SWEEP:
            break;
        default:
            dbgassert(false, "Not implementend signal type");

    }

    feTRACE(15);
}



//////////////////////////////////////////////////////////////////////
