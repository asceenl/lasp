#define TRACERPLUS (-5)
#include "lasp_slm.h"
#include "lasp_assert.h"
#include "lasp_tracer.h"

typedef struct Slm {
    Sosfilterbank *prefilter;  /// Pre-filter, A, or C. If NULL, not used.
    Sosfilterbank *bandpass;   /// Filterbank. If NULL, not used
    Sosfilterbank **splowpass; /// Used for time-weighting of the squared signal
    d ref_level;               /// Reference value for computing decibels
    us downsampling_fac;       /// Every x'th sample is returned.
    us cur_offset;             /// Storage for offset point in input arrays
    vd Pm;    /// Storage for the computing the mean of the square of the signal.
    vd Pmax;  /// Storage for maximum computed signal power so far.
    vd Ppeak; /// Storage for computing peak powers so far.
    us N;     /// Counter for the number of time samples counted that came in

} Slm;

Slm *Slm_create(Sosfilterbank *prefilter, Sosfilterbank *bandpass, const d fs,
        const d tau, const d ref_level, us *downsampling_fac) {
    fsTRACE(15);
    assertvalidptr(downsampling_fac);

    Slm *slm = NULL;
    if (ref_level <= 0) {
        WARN("Invalid reference level");
        return NULL;
    } else if (fs <= 0) {
        WARN("Invalid sampling frequency");
        return NULL;
    }

    slm = (Slm *)a_malloc(sizeof(Slm));
    slm->ref_level = ref_level;
    slm->prefilter = prefilter;
    slm->bandpass = bandpass;

    /// Compute the downsampling factor. This one is chosen based on the
    /// lowpass filter. Which has a -3 dB point of f = 1/(tau*2*pi). See LASP
    /// documentation for the computation of its minus 20 dB point. We set the
    /// reduction in its 'sampling frequency' such that its noise is at a level
    /// of 20 dB less than its 'signal'.
    us ds_fac;
    if (tau > 0) {
        // A reasonable 'framerate' for the sound level meter, based on the
        // filtering time constant.
        d fs_slm = 10 / tau;
        dVARTRACE(15, fs_slm);
        if(fs_slm < 30) {
            fs_slm = 30;
        }
        ds_fac = (us)(fs / fs_slm);
        if (ds_fac == 0) {
            // If we get 0, it should be 1
            ds_fac++;
        }
    } else {
        ds_fac = 1;
    }
    slm->downsampling_fac = ds_fac;
    *downsampling_fac = ds_fac;
    slm->cur_offset = 0;

    /// Create the single pole lowpass
    us filterbank_size;
    if (bandpass) {
        filterbank_size = Sosfilterbank_getFilterbankSize(bandpass);
    } else {
        filterbank_size = 1;
    }

    if (tau > 0) {

        vd lowpass_sos = vd_alloc(6);
        d b0 = 1.0 / (1 + 2 * tau * fs);
        *getvdval(&lowpass_sos, 0) = b0;
        *getvdval(&lowpass_sos, 1) = b0;
        *getvdval(&lowpass_sos, 2) = 0;
        *getvdval(&lowpass_sos, 3) = 1;
        *getvdval(&lowpass_sos, 4) = (1 - 2 * tau * fs) * b0;
        *getvdval(&lowpass_sos, 5) = 0;

        slm->splowpass = a_malloc(filterbank_size * sizeof(Sosfilterbank *));
        for (us ch = 0; ch < filterbank_size; ch++) {
            /// Allocate a filterbank with one channel and one section.
            slm->splowpass[ch] = Sosfilterbank_create(1, 1);
            Sosfilterbank_setFilter(slm->splowpass[ch], 0, lowpass_sos);
        }
        vd_free(&lowpass_sos);
    } else {
        /// No low-pass filtering. Tau set to zero
        slm->splowpass = NULL;
    }

    /// Initialize statistics gatherers
    slm->Ppeak = vd_alloc(filterbank_size);
    slm->Pmax = vd_alloc(filterbank_size);
    slm->Pm = vd_alloc(filterbank_size);
    slm->N = 0;
    vd_set(&(slm->Ppeak), 0);
    vd_set(&(slm->Pmax), 0);
    vd_set(&(slm->Pm), 0);
    feTRACE(15);
    return slm;
}

dmat Slm_run(Slm *slm, vd *input_data) {
    fsTRACE(15);
    assertvalidptr(slm);
    assert_vx(input_data);

    /// First step: run the input data through the pre-filter
    vd prefiltered;
    if (slm->prefilter)
        prefiltered = Sosfilterbank_filter(slm->prefilter, input_data);
    else {
        prefiltered = dmat_foreign(input_data);
    }
    dmat bandpassed;
    if (slm->bandpass) {
        bandpassed = Sosfilterbank_filter(slm->bandpass, &prefiltered);
    } else {
        bandpassed = dmat_foreign(&prefiltered);
    }
    us filterbank_size = bandpassed.n_cols;

    /// Next step: square all values. We do this in-place. Then we filter for
    /// each channel.
    d ref_level = slm->ref_level;
    d *tmp;

    /// Pre-calculate the size of the output data
    us downsampling_fac = slm->downsampling_fac;
    us samples_bandpassed = bandpassed.n_rows;
    iVARTRACE(15, samples_bandpassed);
    iVARTRACE(15, downsampling_fac);
    us cur_offset = slm->cur_offset;

    /// Compute the number of samples output
    us nsamples_output = samples_bandpassed;
    if (downsampling_fac > 1) {
        nsamples_output = (samples_bandpassed - cur_offset) / downsampling_fac;
        if(nsamples_output > samples_bandpassed) {
            // This means overflow of unsigned number calculations
            nsamples_output = 0;
        }
        while(nsamples_output * downsampling_fac + cur_offset < samples_bandpassed) {
            nsamples_output++;
        }
    }

    iVARTRACE(15, nsamples_output);
    iVARTRACE(15, cur_offset);
    dmat levels = dmat_alloc(nsamples_output, filterbank_size);
    us N, ch;

    for (ch = 0; ch < bandpassed.n_cols; ch++) {
        iVARTRACE(15, ch);
        vd chan = dmat_column(&bandpassed, ch);
        /// Inplace squaring of the signal
        for (us sample = 0; sample < bandpassed.n_rows; sample++) {
            tmp = getdmatval(&bandpassed, sample, ch);
            *tmp = *tmp * *tmp;
            *getvdval(&(slm->Ppeak), ch) = d_max(*getvdval(&(slm->Ppeak), ch), *tmp);
        }

        // Now that all data for the channel is squared, we can run it through
        // the low-pass filter
        cur_offset = slm->cur_offset;

        /// Apply single-pole lowpass filter for current filterbank channel
        TRACE(15, "Start filtering");
        vd power_filtered;
        if (slm->splowpass) {
            power_filtered = Sosfilterbank_filter(slm->splowpass[ch], &chan);
        } else {
            power_filtered = dmat_foreign(&chan);
        }
        TRACE(15, "Filtering done");
        dbgassert(chan.n_rows == power_filtered.n_rows, "BUG");

        /// Output resulting levels at a lower interval
        us i = 0;
        N = slm->N;
        d *Pm = getvdval(&(slm->Pm), ch);
        while (cur_offset < samples_bandpassed) {
            iVARTRACE(10, i);
            iVARTRACE(10, cur_offset);

            /// Filtered power.
            const d P = *getvdval(&power_filtered, cur_offset);
            dVARTRACE(15, P);

            /// Compute maximum, compare to current maximum
            *getvdval(&(slm->Pmax), ch) = d_max(*getvdval(&(slm->Pmax), ch), P);

            /// Update mean power
            d Nd = (d) N;
            *Pm = (*Pm*Nd + P ) / (Nd+1);
            N++;
            dVARTRACE(15, *Pm);

            /// Compute level
            d level = 10 * d_log10((P + d_epsilon ) / ref_level / ref_level);

            *getdmatval(&levels, i++, ch) = level;
            cur_offset = cur_offset + downsampling_fac;
        }
        iVARTRACE(15, cur_offset);
        iVARTRACE(15, i);
        dbgassert(i == (int) nsamples_output, "BUG");

        vd_free(&power_filtered);
        vd_free(&chan);
    }
    /// Update sample counter
    dbgassert(ch >0, "BUG");
    slm->N = N;
    slm->cur_offset = cur_offset - samples_bandpassed;

    vd_free(&prefiltered);
    dmat_free(&bandpassed);
    feTRACE(15);
    return levels;
}


static inline vd levels_from_power(const vd* power,const d ref_level){
    fsTRACE(15);

    vd levels = dmat_alloc_from_dmat(power);
    for(us i=0; i< levels.n_rows; i++) {
        *getvdval(&levels, i) = 10 * d_log10(
                (*getvdval(power, i) + d_epsilon) / ref_level / ref_level);

    }
    feTRACE(15);
    return levels;
}

vd Slm_Lpeak(Slm* slm) {
    fsTRACE(15);
    assertvalidptr(slm);
    vd Lpeak = levels_from_power(&(slm->Ppeak), slm->ref_level);
    feTRACE(15);
    return Lpeak;
}

vd Slm_Lmax(Slm* slm) {
    fsTRACE(15);
    assertvalidptr(slm);
    vd Lmax = levels_from_power(&(slm->Pmax), slm->ref_level);
    feTRACE(15);
    return Lmax;
}

vd Slm_Leq(Slm* slm) {
    fsTRACE(15);
    assertvalidptr(slm);
    print_vd(&(slm->Pm));
    vd Leq = levels_from_power(&(slm->Pm), slm->ref_level);
    feTRACE(15);
    return Leq;
}

void Slm_free(Slm *slm) {
    fsTRACE(15);
    assertvalidptr(slm);
    if (slm->prefilter) {
        Sosfilterbank_free(slm->prefilter);
    }

    us filterbank_size;
    if (slm->bandpass) {
        filterbank_size = Sosfilterbank_getFilterbankSize(slm->bandpass);
        Sosfilterbank_free(slm->bandpass);
    } else {
        filterbank_size = 1;
    }
    if (slm->splowpass) {
        for (us ch = 0; ch < filterbank_size; ch++) {
            Sosfilterbank_free(slm->splowpass[ch]);
        }
        a_free(slm->splowpass);
    }
    vd_free(&(slm->Ppeak));
    vd_free(&(slm->Pmax));
    vd_free(&(slm->Pm));
    a_free(slm);

    feTRACE(15);
}
