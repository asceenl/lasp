include "config.pxi"

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
        cdef cmat r = cmat_foreign(result.shape[0],
                                   result.shape[1],
                                   &result_view[0,0])

        cdef dmat t = dmat_foreign(timedata.shape[0],
                                   timedata.shape[1],
                                   &timedata[0,0])

        Fft_fft(self._fft,&t,&r)

        dmat_free(&t)
        cmat_free(&r)

        return result

    def ifft(self,c[::1,:] freqdata):

        cdef us nfft = Fft_nfft(self._fft)
        cdef us nchannels = freqdata.shape[1]
        assert freqdata.shape[0] == nfft//2+1


        # result[:,:] = np.nan+1j*np.nan

        cdef cmat f = cmat_foreign(freqdata.shape[0],
                                   freqdata.shape[1],
                                   &freqdata[0,0])

        timedata = np.empty((nfft,nchannels),
                            dtype=NUMPY_FLOAT_TYPE,
                            order='F')

        cdef d[::1,:] timedata_view = timedata
        cdef dmat t = dmat_foreign(timedata.shape[0],
                                   timedata.shape[1],
                                   &timedata_view[0,0])

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
                             cmat * result)


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


        td = dmat_foreign(nfft,
                          nchannels,
                          &timedata[0,0])

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

        result_mat = cmat_foreign(nfft//2+1,
                                  nchannels*nchannels,
                                  &result_view[0,0,0])



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
                                           const WindowType wt)

    cmat* AvPowerSpectra_addTimeData(const c_AvPowerSpectra* ps,
                                     const dmat * timedata)


    void AvPowerSpectra_free(c_AvPowerSpectra*)
    us AvPowerSpectra_getAverages(const c_AvPowerSpectra*);

cdef class AvPowerSpectra:
    cdef:
        c_AvPowerSpectra* aps
        us nfft, nchannels

    def __cinit__(self,us nfft,
                  us nchannels,
                  d overlap_percentage,
                  us window=rectangular):
        self.aps = AvPowerSpectra_alloc(nfft,
                                        nchannels,
                                        overlap_percentage,
                                        <WindowType> window)
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

        td = dmat_foreign(nsamples,
                          nchannels,
                          &timedata[0,0])

        result_ptr = AvPowerSpectra_addTimeData(self.aps,
                                                &td)

        # The array here is created in such a way that the strides
        # increase with increasing dimension. This is required for
        # interoperability with the C-code, that stores all
        # cross-spectra in a 2D matrix, where the first axis is the
        # frequency axis, and the second axis corresponds to a certain
        # cross-spectrum, as C_ij(f) = result[freq,i+j*nchannels]

        result = np.empty((self.nfft//2+1,nchannels,nchannels),
                          dtype = NUMPY_COMPLEX_TYPE,
                          order='F')
        cdef c[::1,:,:] result_view = result
        cdef cmat res = cmat_foreign(self.nfft//2+1,
                                     nchannels*nchannels,
                                     &result_view[0,0,0])
        # Copy result
        cmat_copy(&res,result_ptr)

        cmat_free(&res)
        dmat_free(&td)

        return result

cdef extern from "lasp_filterbank.h":
    ctypedef struct c_FilterBank "FilterBank"
    c_FilterBank* FilterBank_create(const dmat* h,const us nfft) nogil
    dmat FilterBank_filter(c_FilterBank* fb,const vd* x) nogil
    void FilterBank_free(c_FilterBank* fb) nogil


cdef class FilterBank:
    cdef:
        c_FilterBank* fb
    def __cinit__(self,d[::1,:] h, us nfft):
        cdef dmat hmat = dmat_foreign(h.shape[0],h.shape[1],&h[0,0])
        self.fb = FilterBank_create(&hmat,nfft)
        dmat_free(&hmat)
        if not self.fb:
            raise RuntimeError('Error creating FilberBank')

    def __dealloc__(self):
        if self.fb:
            FilterBank_free(self.fb)

    def filter_(self,d[:] input_):
        cdef vd input_vd = vd_foreign(input_.size,&input_[0])
        cdef dmat output = FilterBank_filter(self.fb,&input_vd)

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
        cdef dmat d_samples = dmat_foreign(samples.shape[0],
                                           samples.shape[1],
                                           &samples[0,0])
        
        cdef dmat res = Decimator_decimate(self.dec,&d_samples)
        result = dmat_to_ndarray(&res,True)
        dmat_free(&res)
        return result
        
    def __dealloc__(self):
        if self.dec != NULL:
            Decimator_free(self.dec)