"""
This file contains the Cython wrapper functions to C implementations.
"""
include "config.pxi"
IF LASP_FLOAT == "double":
    ctypedef  double d
    ctypedef double complex c
    NUMPY_FLOAT_TYPE = np.float64
    NUMPY_COMPLEX_TYPE = np.complex128
    CYTHON_NUMPY_FLOAT_t = cnp.NPY_FLOAT64
    CYTHON_NUMPY_COMPLEX_t = cnp.NPY_COMPLEX128

ELSE:
    ctypedef  float d
    ctypedef float complex c
    NUMPY_FLOAT_TYPE = np.float32
    NUMPY_COMPLEX_TYPE = np.complex64
    CYTHON_NUMPY_FLOAT_t = cnp.NPY_FLOAT32
    CYTHON_NUMPY_COMPLEX_t = cnp.NPY_COMPLEX64

ctypedef size_t us

cdef extern from "lasp_tracer.h":
    void setTracerLevel(int)
    void TRACE(int,const char*)
    void fsTRACE(int)
    void feTRACE(int)
    void clearScreen()


cdef extern from "lasp_mat.h" nogil:
    ctypedef struct dmat:
        us n_cols
        us n_rows
        d* _data
        bint _foreign_data
    ctypedef struct cmat:
        pass
    ctypedef cmat vc
    ctypedef dmat vd

    dmat dmat_foreign_data(us n_rows,
                           us n_cols,
                           d* data,
                           bint own_data)
    cmat cmat_foreign_data(us n_rows,
                           us n_cols,
                           c* data,
                           bint own_data)

    cmat cmat_alloc(us n_rows,us n_cols)
    dmat dmat_alloc(us n_rows,us n_cols)
    vd vd_foreign(const us size,d* data)
    void vd_free(vd*)

    void dmat_free(dmat*)
    void cmat_free(cmat*)
    void cmat_copy(cmat* to,cmat* from_)


cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)


cdef extern from "lasp_python.h":
    object dmat_to_ndarray(dmat*,bint transfer_ownership)

__all__ = ['AvPowerSpectra']

setTracerLevel(15)
cdef extern from "cblas.h":
    int openblas_get_num_threads()
    void openblas_set_num_threads(int)


# If we touch this variable: we get segfaults when running from
# Spyder!
# openblas_set_num_threads(8)
# print("Number of threads: ",
# openblas_get_num_threads())

def cls():
    clearScreen()
# cls()

cdef extern from "lasp_fft.h":
    ctypedef struct c_Fft "Fft"
    c_Fft* Fft_create(us nfft)
    void Fft_free(c_Fft*)
    void Fft_fft(c_Fft*,dmat * timedate,cmat * res) nogil
    void Fft_ifft(c_Fft*,cmat * freqdata,dmat* timedata) nogil
    us Fft_nfft(c_Fft*)



cdef class Fft:
    cdef:
        c_Fft* _fft

    def __cinit__(self, us nfft):
        self._fft = Fft_create(nfft)
        if self._fft == NULL:
            raise RuntimeError('Fft allocation failed')

    def __dealloc__(self):
        if self._fft!=NULL:
            Fft_free(self._fft)

    def fft(self,d[::1,:] timedata):

        cdef us nfft = Fft_nfft(self._fft)
        cdef us nchannels = timedata.shape[1]
        assert timedata.shape[0] ==nfft

        result = np.empty((nfft//2+1,nchannels),
                          dtype=NUMPY_COMPLEX_TYPE,
                          order='F')

        # result[:,:] = np.nan+1j*np.nan
        cdef c[::1,:] result_view = result
        cdef cmat r = cmat_foreign_data(result.shape[0],
                                        result.shape[1],
                                        &result_view[0,0],
                                        False)

        cdef dmat t = dmat_foreign_data(timedata.shape[0],
                                        timedata.shape[1],
                                        &timedata[0,0],
                                        False)
        with nogil:
            Fft_fft(self._fft,&t,&r)

            dmat_free(&t)
            cmat_free(&r)

        return result

    def ifft(self,c[::1,:] freqdata):

        cdef us nfft = Fft_nfft(self._fft)
        cdef us nchannels = freqdata.shape[1]
        assert freqdata.shape[0] == nfft//2+1


        # result[:,:] = np.nan+1j*np.nan

        cdef cmat f = cmat_foreign_data(freqdata.shape[0],
                                        freqdata.shape[1],
                                        &freqdata[0,0],
                                        False)

        timedata = np.empty((nfft,nchannels),
                            dtype=NUMPY_FLOAT_TYPE,
                            order='F')

        cdef d[::1,:] timedata_view = timedata
        cdef dmat t = dmat_foreign_data(timedata.shape[0],
                                        timedata.shape[1],
                                        &timedata_view[0,0],
                                        False)

        with nogil:
            Fft_ifft(self._fft,&f,&t)

            dmat_free(&t)
            cmat_free(&f)

        return timedata

cdef extern from "lasp_window.h":
    ctypedef enum WindowType:
        Hann
        Hamming
        Rectangular
        Bartlett
        Blackman

# Export these constants to Python
class Window:
    hann = Hann
    hamming = Hamming
    rectangular = Rectangular
    bartlett = Bartlett
    blackman = Blackman

cdef extern from "lasp_ps.h":
    ctypedef struct c_PowerSpectra "PowerSpectra"
    c_PowerSpectra* PowerSpectra_alloc(const us nfft,
                                       const WindowType wt)

    void PowerSpectra_compute(const c_PowerSpectra* ps,
                             const dmat * timedata,
                             cmat * result) nogil


    void PowerSpectra_free(c_PowerSpectra*)

cdef class PowerSpectra:
    cdef:
        c_PowerSpectra* _ps

    def __cinit__(self, us nfft,us window=Window.rectangular):
        self._ps = PowerSpectra_alloc(nfft,<WindowType> window)
        if self._ps == NULL:
            raise RuntimeError('PowerSpectra allocation failed')

    def compute(self,d[::1,:] timedata):
        cdef:
            us nchannels = timedata.shape[1]
            us nfft = timedata.shape[0]
            int rv
            dmat td
            cmat result_mat


        td = dmat_foreign_data(nfft,
                               nchannels,
                               &timedata[0,0],
                               False)

        # The array here is created in such a way that the strides
        # increase with increasing dimension. This is required for
        # interoperability with the C-code, that stores all
        # cross-spectra in a 2D matrix, where the first axis is the
        # frequency axis, and the second axis corresponds to a certain
        # cross-spectrum, as C_ij(f) = result[freq,i+j*nchannels]

        result = np.empty((nfft//2+1,nchannels,nchannels),
                          dtype = NUMPY_COMPLEX_TYPE,
                          order='F')

        cdef c[::1,:,:] result_view = result

        result_mat = cmat_foreign_data(nfft//2+1,
                                       nchannels*nchannels,
                                       &result_view[0,0,0],
                                       False)


        with nogil:
            PowerSpectra_compute(self._ps,&td,&result_mat)

        dmat_free(&td)
        cmat_free(&result_mat)

        return result


    def __dealloc__(self):
        if self._ps != NULL:
            PowerSpectra_free(self._ps)


cdef extern from "lasp_aps.h":
    ctypedef struct c_AvPowerSpectra "AvPowerSpectra"
    c_AvPowerSpectra* AvPowerSpectra_alloc(const us nfft,
                                           const us nchannels,
                                           d overlap_percentage,
                                           const WindowType wt,
                                           const vd* weighting)

    cmat* AvPowerSpectra_addTimeData(const c_AvPowerSpectra* ps,
                                     const dmat * timedata) nogil


    void AvPowerSpectra_free(c_AvPowerSpectra*)
    us AvPowerSpectra_getAverages(const c_AvPowerSpectra*);

cdef class AvPowerSpectra:
    cdef:
        c_AvPowerSpectra* aps
        us nfft, nchannels

    def __cinit__(self,us nfft,
                  us nchannels,
                  d overlap_percentage,
                  us window=Window.hann,
                  d[:] weighting = np.array([])):


        cdef vd weighting_vd
        cdef vd* weighting_ptr = NULL
        if(weighting.size != 0):
            weighting_vd = dmat_foreign_data(weighting.size,1,
                                             &weighting[0],False)
            weighting_ptr = &weighting_vd

        self.aps = AvPowerSpectra_alloc(nfft,
                                        nchannels,
                                        overlap_percentage,
                                        <WindowType> window,
                                        weighting_ptr)
        self.nchannels = nchannels
        self.nfft = nfft

        if self.aps == NULL:
            raise RuntimeError('AvPowerSpectra allocation failed')

    def __dealloc__(self):
        if self.aps:
            AvPowerSpectra_free(self.aps)
    def getAverages(self):
        return AvPowerSpectra_getAverages(self.aps)

    def addTimeData(self,d[::1,:] timedata):
        """!
        Adds time data, returns current result
        """
        cdef:
            us nsamples = timedata.shape[0]
            us nchannels = timedata.shape[1]
            dmat td
            cmat* result_ptr

        if nchannels != self.nchannels:
            raise RuntimeError('Invalid number of channels')

        td = dmat_foreign_data(nsamples,
                               nchannels,
                               &timedata[0,0],
                               False)

        result = np.empty((self.nfft//2+1,nchannels,nchannels),
                          dtype = NUMPY_COMPLEX_TYPE,
                          order='F')

        cdef c[::1,:,:] result_view = result

        cdef cmat res = cmat_foreign_data(self.nfft//2+1,
                                          nchannels*nchannels,
                                          &result_view[0,0,0],
                                          False)
        with nogil:
            result_ptr = AvPowerSpectra_addTimeData(self.aps,
                                                &td)

        # The array here is created in such a way that the strides
        # increase with increasing dimension. This is required for
        # interoperability with the C-code, that stores all
        # cross-spectra in a 2D matrix, where the first axis is the
        # frequency axis, and the second axis corresponds to a certain
        # cross-spectrum, as C_ij(f) = result[freq,i+j*nchannels]

            # Copy result
            cmat_copy(&res,result_ptr)

            cmat_free(&res)
            dmat_free(&td)

        return result

cdef extern from "lasp_firfilterbank.h":
    ctypedef struct c_Firfilterbank "Firfilterbank"
    c_Firfilterbank* Firfilterbank_create(const dmat* h,const us nfft) nogil
    dmat Firfilterbank_filter(c_Firfilterbank* fb,const vd* x) nogil
    void Firfilterbank_free(c_Firfilterbank* fb) nogil


cdef class FilterBank:
    cdef:
        c_Firfilterbank* fb
    def __cinit__(self,d[::1,:] h, us nfft):
        cdef dmat hmat = dmat_foreign_data(h.shape[0],
                                           h.shape[1],
                                           &h[0,0],
                                           False)

        self.fb = Firfilterbank_create(&hmat,nfft)
        dmat_free(&hmat)
        if not self.fb:
            raise RuntimeError('Error creating FilberBank')

    def __dealloc__(self):
        if self.fb:
            Firfilterbank_free(self.fb)

    def filter_(self,d[::1, :] input_):
        assert input_.shape[1] == 1
        cdef dmat input_vd = dmat_foreign_data(input_.shape[0],1,
                                             &input_[0, 0],False)

        
        cdef dmat output
        with nogil:
            output = Firfilterbank_filter(self.fb,&input_vd)

        # Steal the pointer from output
        result = dmat_to_ndarray(&output,True)

        dmat_free(&output)
        vd_free(&input_vd)

        return result

cdef extern from "lasp_decimation.h":
    ctypedef struct c_Decimator "Decimator"
    ctypedef enum DEC_FAC:
        DEC_FAC_4

    c_Decimator* Decimator_create(us nchannels,DEC_FAC d) nogil
    dmat Decimator_decimate(c_Decimator* dec,const dmat* samples) nogil
    void Decimator_free(c_Decimator* dec) nogil

cdef class Decimator:
    cdef:
        c_Decimator* dec
        us nchannels
    def __cinit__(self, us nchannels,us dec_fac):
        assert dec_fac == 4, 'Invalid decimation factor'
        self.nchannels = nchannels
        self.dec = Decimator_create(nchannels,DEC_FAC_4)
        if not self.dec:
            raise RuntimeError('Error creating decimator')

    def decimate(self,d[::1,:] samples):
        assert samples.shape[1] == self.nchannels,'Invalid number of channels'
        if samples.shape[0] == 0:
            return np.zeros((0, self.nchannels))

        cdef dmat d_samples = dmat_foreign_data(samples.shape[0],
                                                samples.shape[1],
                                                &samples[0,0],
                                                False)

        cdef dmat res = Decimator_decimate(self.dec,&d_samples)
        result = dmat_to_ndarray(&res,True)
        dmat_free(&res)
        return result

    def __dealloc__(self):
        if self.dec != NULL:
            Decimator_free(self.dec)

cdef extern from "lasp_sp_lowpass.h":
    ctypedef struct c_SPLowpass "SPLowpass"
    c_SPLowpass* SPLowpass_create(d fs,d tau)
    vd SPLowpass_filter(c_SPLowpass* lp,
                        const vd* _input)
    void SPLowpass_free(c_SPLowpass* lp)

cdef class SPLowpass:
    cdef:
        c_SPLowpass* lp
    def __cinit__(self,d fs,d tau):
        self.lp = SPLowpass_create(fs,tau)
        if not self.lp:
            raise RuntimeError('Error creating lowpass filter')

    def __dealloc__(self):
        if self.lp:
            SPLowpass_free(self.lp)

    def filter_(self,d[::1,:] input_):
        assert input_.shape[1] == 1
        if input_.shape[0] == 0:
            return np.array([],dtype=NUMPY_FLOAT_TYPE)

        cdef vd input_vd = dmat_foreign_data(input_.shape[0],1,
                                             &input_[0,0],False)

        cdef dmat output = SPLowpass_filter(self.lp,&input_vd)

        # # Steal the pointer from output
        result = dmat_to_ndarray(&output,True)

        dmat_free(&output)
        vd_free(&input_vd)

        return result


cdef extern from "lasp_siggen.h":
    ctypedef struct c_Siggen "Siggen"
    c_Siggen* Siggen_Whitenoise_create(d fs, d level_dB)
    c_Siggen* Siggen_Sinewave_create(d fs, d freq, d level_dB)
    void Siggen_genSignal(c_Siggen*, vd* samples) nogil
    void Siggen_free(c_Siggen*)


cdef class Siggen:
    cdef c_Siggen *_siggen

    def __cinit__(self):
        self._siggen = NULL

    def __dealloc__(self):
        if self._siggen:
            Siggen_free(self._siggen)

    def genSignal(self, us nsamples):
        output = np.empty(nsamples, dtype=np.float)
        assert self._siggen != NULL

        cdef d[:] output_view = output

        cdef dmat output_dmat = dmat_foreign_data(nsamples,
                                                  1,
                                                  &output_view[0],
                                                  False)
        with nogil:
            Siggen_genSignal(self._siggen,
                             &output_dmat)

        return output


    @staticmethod
    def sineWave(d fs,d freq,d level_dB):
        cdef c_Siggen* c_siggen = Siggen_Sinewave_create(fs, freq, level_dB)
        siggen = Siggen()
        siggen._siggen = c_siggen
        return siggen


    @staticmethod
    def whiteNoise(d fs, d level_dB):
        cdef c_Siggen* c_siggen = Siggen_Whitenoise_create(fs, level_dB)
        siggen = Siggen()
        siggen._siggen = c_siggen
        return siggen
