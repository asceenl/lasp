include "config.pxi"

cdef extern from "fft.h":
    ctypedef struct c_Fft "Fft"
    c_Fft* Fft_alloc(us nfft,us nchannels)
    void Fft_free(c_Fft*)
    void Fft_fft(c_Fft*,dmat * timedate,cmat * res) nogil
    us Fft_nchannels(c_Fft*)
    us Fft_nfft(c_Fft*)

cdef class Fft:
    cdef:
        c_Fft* _fft

    def __cinit__(self, us nfft,us nchannels):
        self._fft = Fft_alloc(nfft,nchannels)
        if self._fft == NULL:
            raise RuntimeError('Fft allocation failed')

    def __dealloc__(self):
        if self._fft!=NULL:
            Fft_free(self._fft)

    def fft(self,d[::1,:] timedata):

        cdef us nfft = Fft_nfft(self._fft)
        cdef us nchannels = Fft_nchannels(self._fft)
        assert timedata.shape[0] ==nfft
        assert timedata.shape[1] == nchannels
        
        result = np.empty((nfft//2+1,
                           nchannels),
                           dtype=NUMPY_COMPLEX_TYPE,order='F')
        # result[:,:] = np.nan+1j*np.nan

        cdef cmat r = cmat_from_array(result)
        cdef dmat t = dmat_from_array(timedata)
        Fft_fft(self._fft,&t,&r)

        return result


cdef extern from "window.h":
    ctypedef enum WindowType:
        Hann = 0
        Hamming = 1
        Blackman = 2
        Rectangular = 3

cdef extern from "ps.h":
    ctypedef struct c_PowerSpectra "PowerSpectra"
    c_PowerSpectra* PowerSpectra_alloc(const us nfft,
                                     const us nchannels,
                                     const WindowType wt)
    int PowerSpectra_compute(const c_PowerSpectra* ps,
                             const dmat * timedata,
                             cmat * result)
    

    void PowerSpectra_free(c_PowerSpectra*)

# cdef class PowerSpectra:
#     cdef:
#         c_PowerSpectra* _ps

#     def __cinit__(self, us nfft,us nchannels):
#         self._ps = PowerSpectra_alloc(nfft,nchannels,Rectangular)
#         if self._ps == NULL:
#             raise RuntimeError('PowerSpectra allocation failed')

#     def compute(self,d[::1,:] timedata):
#         cdef:
#             us nchannels = timedata.shape[1]
#             us nfft = timedata.shape[0]
#             int rv
#             dmat td
#             cmat result_mat
#         td.data = &timedata[0,0]
#         td.n_rows = nfft
#         td.n_cols = nchannels
#         result = np.empty((nfft//2+1,nchannels*nchannels),
#                           dtype = NUMPY_COMPLEX_TYPE,
#                           order='F')
#         result_mat = cmat_from_array(result)

#         rv = PowerSpectra_compute(self._ps,&td,&result_mat)
#         if rv !=0:
#             raise RuntimeError("Error computing power spectra")

#         return result


#     def __dealloc__(self):
#         if self._ps != NULL:
#             PowerSpectra_free(self._ps)
