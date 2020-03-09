// lasp_siggen.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
// Signal generator implementation
//////////////////////////////////////////////////////////////////////
#define TRACERPLUS (-5)
#include "lasp_siggen.h"
#include "lasp_alloc.h"
#include "lasp_assert.h"
#include "lasp_mat.h"

#define PRIVATE_SIZE 64

typedef enum {
    SINEWAVE = 0,
    NOISE,
    SWEEP,

} SignalType;

typedef struct Siggen {
    SignalType signaltype;
    d fs; // Sampling frequency [Hz]
    d level_amp;
    char private_data[PRIVATE_SIZE];
} Siggen;

typedef struct { 
    d curtime;
    d omg;
} SinewaveSpecific;

typedef struct {
    d fl;
    d fu;
    d Ts;
    d phase;
    d tau;
    bool pos;
    us flags;
} SweepSpecific;

typedef struct {
    d V1, V2, S;
    int phase;
    Sosfilterbank* colorfilter;
} NoiseSpecific;

static d level_amp(d level_dB){
    return pow(10, level_dB/20);
}

Siggen* Siggen_create(SignalType type, const d fs,const d level_dB) {

    fsTRACE(15);

    Siggen* siggen = a_malloc(sizeof(Siggen));
    siggen->signaltype = type;
    siggen->fs = fs;
    siggen->level_amp = level_amp(level_dB);

    feTRACE(15);
    return siggen;
}

Siggen* Siggen_Sinewave_create(const d fs, const d freq,const d level_dB) {
    fsTRACE(15);

    Siggen* sine = Siggen_create(SINEWAVE, fs, level_dB);
    dbgassert(sizeof(SinewaveSpecific) <= sizeof(sine->private_data),
            "Allocated memory too small");
    SinewaveSpecific* sp = (SinewaveSpecific*) sine->private_data;
    sp->curtime = 0;
    sp->omg = 2*number_pi*freq;

    feTRACE(15);
    return sine;
}

Siggen* Siggen_Noise_create(const d fs, const d level_dB, Sosfilterbank* colorfilter) {
    fsTRACE(15);

    Siggen* noise = Siggen_create(NOISE, fs, level_dB);
    dbgassert(sizeof(NoiseSpecific) <= sizeof(noise->private_data),
            "Allocated memory too small");
    NoiseSpecific* wn = (NoiseSpecific*) noise->private_data;
    wn->phase = 0;
    wn->V1 = 0;
    wn->V2 = 0;
    wn->S = 0;
    wn->colorfilter = colorfilter;

    feTRACE(15);
    return noise;
}


Siggen* Siggen_Sweep_create(const d fs,const d fl,const d fu,
        const d Ts, const us flags, const d level_dB) {
    fsTRACE(15);

    Siggen* sweep = Siggen_create(SWEEP, fs, level_dB);
    dbgassert(sizeof(SweepSpecific) <= sizeof(sweep->private_data), 
            "Allocated memory too small");
    // Set pointer to inplace data storage
    SweepSpecific* sp = (SweepSpecific*) sweep->private_data;
    if(fl < 0 || fu < 0 || Ts <= 0) {
        return NULL;
    }

    sp->flags = flags;

    sp->fl = fl;
    sp->fu = fu;
    sp->Ts = Ts;
    sp->phase = 0;
    sp->pos = flags & SWEEP_FLAG_BACKWARD ? false: true;
    if(flags & SWEEP_FLAG_BACKWARD) {
        sp->tau = Ts;
    } else {
        sp->tau = 0;
    }
    /* sp->pos = false; */
    /* sp->tau = Ts/2; */

    feTRACE(15);
    return sweep;
}

void Siggen_free(Siggen* siggen) {
    fsTRACE(15);
    assertvalidptr(siggen);
    NoiseSpecific* sp;

    switch(siggen->signaltype) {
        case SWEEP:
            /* Sweep specific stuff here */
            break;
        case SINEWAVE:
            /* Sweep specific stuff here */
            break;
        case NOISE:
            sp = (NoiseSpecific*) siggen->private_data;
            if(sp->colorfilter) {
                Sosfilterbank_free(sp->colorfilter);
            }

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

static void Sweep_genSignal(Siggen* siggen, SweepSpecific* sweep,
        vd* samples) {
    fsTRACE(15);
    assertvalidptr(sweep);

    const d fl = sweep->fl;
    const d fu = sweep->fu;
    const d deltat = 1/siggen->fs;
    const d Ts = sweep->Ts;

    const d Thalf = Ts/2;

    dVARTRACE(20, deltat);

    // Load state
    d tau = sweep->tau;
    bool pos = sweep->pos;

    // Obtain flags and expand
    us flags = sweep->flags;
    bool forward_sweep = flags & SWEEP_FLAG_FORWARD;
    bool backward_sweep = flags & SWEEP_FLAG_BACKWARD;
    dbgassert(!(forward_sweep && backward_sweep), "Both forward and backward flag set");

    d k, Treverse;
    if(forward_sweep || backward_sweep) {
        k = (fu - fl)/Ts;
        Treverse = Ts;
    }
    else {
        k = (fu - fl)/Thalf;
        Treverse = Ts/2;
    }


    /* const d k = 0; */

    d phase = sweep->phase;
    d curfreq;
    for(us i =0; i< samples->n_rows; i++) {

        curfreq = fl + k*tau;
        phase = phase + 2*number_pi*curfreq*deltat;

        // Subtract some to avoid possible overflow. Don't know whether such a
        // thing really happens
        if(phase > 2*number_pi)
            phase = phase - 2*number_pi;

        if(pos) {
            tau = tau + deltat;
            if(tau >= Treverse) { 
                if(forward_sweep) { tau = 0; }
                else if(backward_sweep) { dbgassert(false, "BUG"); }
                else { pos = false; }
            }

        } else {
            /* dbgassert(false, "cannot get here"); */
            tau = tau - deltat;
            if(tau <= 0) { 
                if(backward_sweep) { tau = Treverse; }
                else if(forward_sweep) { dbgassert(false, "BUG"); }
                else { pos = true; }
            }
        }
        setvecval(samples, i, siggen->level_amp*d_sin(phase));
    }
    // Store state
    sweep->phase = phase;
    sweep->pos = pos;
    sweep->tau = tau;
    feTRACE(15);
}

static void noise_genSignal(Siggen* siggen, NoiseSpecific* wn, vd* samples) {
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

            X = V1 * sqrt(-2 * d_ln(S) / S);
        } else
            X = V2 * sqrt(-2 * d_ln(S) / S);

        phase = 1 - phase;

        setvecval(samples, i, siggen->level_amp*X);
    }
    if(wn->colorfilter){
        vd filtered = Sosfilterbank_filter(wn->colorfilter,
                samples);
        dmat_copy(samples, &filtered);
        vd_free(&filtered);
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

    switch(siggen->signaltype) {
        case SINEWAVE:
            Sinewave_genSignal(siggen,
                    (SinewaveSpecific*) siggen->private_data,
                    samples);

            break;
        case NOISE:
            noise_genSignal(siggen,
                    (NoiseSpecific*) siggen->private_data,
                    samples);
            break;
        case SWEEP:
            Sweep_genSignal(siggen,
                    (SweepSpecific*) siggen->private_data,
                    samples);
            break;
        default:
            dbgassert(false, "Not implementend signal type");

    }

    feTRACE(15);
}



//////////////////////////////////////////////////////////////////////
