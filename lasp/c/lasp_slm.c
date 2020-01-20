#define TRACERPLUS (10)
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
    vd Leq; /// Storage for the computed equivalent levels so far.

} Slm;

Slm *Slm_create(Sosfilterbank *prefilter, Sosfilterbank *bandpass, const d fs,
        const d tau, const d ref_level, us *downsampling_fac) {
    fsTRACE(15);
    assertvalidptr(downsampling_fac);

    Slm *slm = NULL;
    if (tau <= 0) {
        WARN("Invalid time constant");
        return NULL;
    } else if (ref_level <= 0) {
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
    const d fs_slm = 1 / (2 * number_pi * tau) * (1 - 0.01) / 0.01;
    slm->downsampling_fac = (us)(fs / fs_slm);
    slm->cur_offset = 0;
    *downsampling_fac = slm->downsampling_fac;

    /// Create the single pole lowpass
    us filterbank_size;
    if (bandpass) {
        filterbank_size = Sosfilterbank_getFilterbankSize(bandpass);
    } else {
        filterbank_size = 1;
    }

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
    us cur_offset = slm->cur_offset;

    int nsamples_output = (samples_bandpassed - cur_offset) / downsampling_fac;
    while(nsamples_output*downsampling_fac + cur_offset < samples_bandpassed)
        nsamples_output++;
    if(nsamples_output < 0) nsamples_output = 0;

    iVARTRACE(15, nsamples_output);
    iVARTRACE(15, cur_offset);

    dmat levels = dmat_alloc(nsamples_output, filterbank_size);

    for (us ch = 0; ch < bandpassed.n_cols; ch++) {
        iVARTRACE(15, ch);
        /// Inplace squaring of the signal
        for (us sample = 0; sample < bandpassed.n_rows; sample++) {
            tmp = getdmatval(&bandpassed, sample, ch);
            *tmp = *tmp * *tmp;
        }

        // Now that all data for the channel is squared, we can run it through
        // the low-pass filter
        vd chan = dmat_column(&bandpassed, ch);

        /// Apply single-pole lowpass filter for current filterbank channel
        vd power_filtered = Sosfilterbank_filter(slm->splowpass[ch], &chan);
        dbgassert(chan.n_rows == power_filtered.n_rows, "BUG");

        /// Output resulting levels at a lower interval
        us i = 0;
        d level;
        while (cur_offset < samples_bandpassed) {
            iVARTRACE(10, i);
            iVARTRACE(10, cur_offset);
            level = 10 * d_log10(*getvdval(&power_filtered, cur_offset) / ref_level /
                    ref_level);

            *getdmatval(&levels, i++, ch) = level;
            cur_offset = cur_offset + downsampling_fac;
        }
        iVARTRACE(15, cur_offset);
        iVARTRACE(15, i);
        dbgassert(i == nsamples_output, "BUG");
        slm->cur_offset = cur_offset - samples_bandpassed;

        vd_free(&chan);
        vd_free(&power_filtered);
    }

    vd_free(&prefiltered);
    dmat_free(&bandpassed);
    feTRACE(15);
    return levels;
}

void Slm_free(Slm *slm) {
    fsTRACE(15);
    assertvalidptr(slm);
    if (slm->prefilter) {
        Sosfilterbank_free(slm->prefilter);
    }
    assertvalidptr(slm->splowpass);

    us filterbank_size;
    if (slm->bandpass) {
        filterbank_size = Sosfilterbank_getFilterbankSize(slm->bandpass);
        Sosfilterbank_free(slm->bandpass);
    } else {
        filterbank_size = 1;
    }
    for (us ch = 0; ch < filterbank_size; ch++) {
        Sosfilterbank_free(slm->splowpass[ch]);
    }
    a_free(slm->splowpass);

    a_free(slm);

    feTRACE(15);
}
