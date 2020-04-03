import sys
include "config.pxi"
cimport cython
from .lasp_daqconfig import DeviceInfo
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf, fprintf, stderr
from libc.string cimport memcpy, memset
from cpython.ref cimport PyObject,Py_INCREF, Py_DECREF


cdef extern from "RtAudio.h" nogil:
    ctypedef unsigned long RtAudioStreamStatus
    RtAudioStreamStatus RTAUDIO_INPUT_OVERFLOW
    RtAudioStreamStatus RTAUDIO_OUTPUT_UNDERFLOW

    cdef cppclass RtAudioError:
        ctypedef enum Type:
            WARNING
            DEBUG_WARNING
            UNSPECIFIED
            NO_DEVICES_FOUND
            INVALID_DEVICE
            MEMORY_ERROR
            INVALID_PARAMETER
            INVALID_USE
            DRIVER_ERROR
            SYSTEM_ERROR
            THREAD_ERROR

    ctypedef unsigned long RtAudioStreamFlags
    RtAudioStreamFlags RTAUDIO_NONINTERLEAVED
    RtAudioStreamFlags RTAUDIO_MINIMIZE_LATENCY
    RtAudioStreamFlags RTAUDIO_HOG_DEVICE
    RtAudioStreamFlags RTAUDIO_SCHEDULE_REALTIME
    RtAudioStreamFlags RTAUDIO_ALSA_USE_DEFAULT
    RtAudioStreamFlags RTAUDIO_JACK_DONT_CONNECT

    ctypedef unsigned long RtAudioFormat
    RtAudioFormat RTAUDIO_SINT8
    RtAudioFormat RTAUDIO_SINT16
    RtAudioFormat RTAUDIO_SINT24
    RtAudioFormat RTAUDIO_SINT32
    RtAudioFormat RTAUDIO_FLOAT32
    RtAudioFormat RTAUDIO_FLOAT64

    ctypedef int (*RtAudioCallback)(void* outputBuffer,
                                    void* inputBuffer,
                                    unsigned int nFrames,
                                    double streamTime,
                                    RtAudioStreamStatus status,
                                    void* userData)

    ctypedef void (*RtAudioErrorCallback)(RtAudioError.Type _type,
                                          const string& errortxt)

    cdef cppclass cppRtAudio "RtAudio":
        cppclass DeviceInfo:
            bool probed
            string name
            unsigned int outputChannels
            unsigned int inputChannels
            unsigned int duplexChannels
            bool isDefaultOutput
            bool isDefaultInput
            vector[unsigned int] sampleRates
            unsigned int preferredSampleRate
            RtAudioFormat nativeFormats

        cppclass StreamOptions:
            RtAudioStreamFlags flags
            unsigned int numberOfBuffers
            string streamName
            int priority
        cppclass StreamParameters:
            unsigned int deviceId
            unsigned int nChannels
            unsigned int firstChannel

        RtAudio() except +
        # ~RtAudio() Destructors should not be listed
        unsigned int getDeviceCount()
        DeviceInfo getDeviceInfo(unsigned int device)
        unsigned int getDefaultOutputDevice()
        unsigned int getDefaultInputDevice()
        void openStream(StreamParameters* outputParameters,
                        StreamParameters* intputParameters,
                        RtAudioFormat _format,
                        unsigned int sampleRate,
                        unsigned int* bufferFrames,
                        RtAudioCallback callback,
                        void* userData,
                        void* StreamOptions,
                        RtAudioErrorCallback) except +
        void closeStream()
        void startStream() except +
        void stopStream() except +
        void abortStream() except +
        bool isStreamOpen()
        bool isStreamRunning()
        double getStreamTime()
        void setStreamTime(double) except +
        long getStreamLatency()
        unsigned int getStreamSampleRate()
        void showWarnings(bool value)
    
cdef extern from "lasp_cppthread.h" nogil:
    cdef cppclass CPPThread[T,F]:
        CPPThread(F threadfunction, T data)
        void join()

    void CPPsleep(unsigned int ms)

cdef extern from "lasp_cppqueue.h" nogil:
    cdef cppclass SafeQueue[T]:
        SafeQueue()
        void enqueue(T t)
        T dequeue()
        size_t size() const
        bool empty() const


cdef extern from "atomic" namespace "std" nogil:
    cdef cppclass atomic[T]:
        T load()
        void store(T)

_formats_strkey = {
        '8-bit integers': (RTAUDIO_SINT8,   1, np.int8),
        '16-bit integers': (RTAUDIO_SINT16, 2, np.int16),
        '24-bit integers': (RTAUDIO_SINT24, 3),
        '32-bit integers': (RTAUDIO_SINT32, 4, np.int32),
        '32-bit floats': (RTAUDIO_FLOAT32,  4, np.float32),
        '64-bit floats': (RTAUDIO_FLOAT64,  8, np.float64),
}
_formats_rtkey = {
        RTAUDIO_SINT8: ('8-bit integers', 1, cnp.NPY_INT8),
        RTAUDIO_SINT16: ('16-bit integers',2, cnp.NPY_INT16),
        RTAUDIO_SINT24: ('24-bit integers',3),
        RTAUDIO_SINT32: ('32-bit integers',4, cnp.NPY_INT32),
        RTAUDIO_FLOAT32: ('32-bit floats',  4, cnp.NPY_FLOAT32),
        RTAUDIO_FLOAT64: ('64-bit floats',  8, cnp.NPY_FLOAT64),
}

def get_numpy_dtype_from_format_string(format_string):
    return _formats_strkey[format_string][-1]
def get_sampwidth_from_format_string(format_string):
    return _formats_strkey[format_string][-2]



# It took me quite a long time to fully understand Cython's idiosyncrasies
# concerning C(++) callbacks, the GIL and passing Python objects as pointers
# into C(++) functions. But finally, here it is!
cdef object fromBufferToNPYNoCopy(
                             cnp.NPY_TYPES buffer_format_type,
                             void* buf,
                             size_t nchannels,
                             size_t nframes):
    cdef cnp.npy_intp[2] dims = [nframes, nchannels]
 
    # Interleaved data is C-style contiguous. Therefore, we can directly use
    # SimpleNewFromData()
    array = cnp.PyArray_SimpleNewFromData(2, &dims[0], buffer_format_type,
            buf)

    return array


cdef void fromNPYToBuffer(cnp.ndarray arr,
                          void* buf):
    """
    Copy a Python numpy array over to a buffer
    No checks, just memcpy! Careful!
    """
    memcpy(buf, arr.data, arr.size*arr.itemsize)

ctypedef struct _Stream:
    PyObject* pyCallback
    RtAudioFormat sampleformat

    atomic[bool] stopThread

    unsigned int nFrames

    cppRtAudio.StreamParameters inputParams
    cppRtAudio.StreamParameters outputParams

    # If these queue pointers are NULL, it means the stream does not have an
    # input, or output.
    SafeQueue[void*] *inputQueue
    SafeQueue[void*] *outputQueue
    size_t inputbuffersize  # Full size of the output buffer, in BYTES
    size_t outputbuffersize # Full size of the output buffer, in BYTES
    CPPThread[void*, void (*)(void*)] *thread

    
cdef int audioCallback(void* outputbuffer,
                       void* inputbuffer,
                       unsigned int nFrames,
                       double streamTime,
                       RtAudioStreamStatus status,
                       void* userData) nogil:
    """
    Calls the Python callback function and converts data

    """
    cdef:
        int rval = 0
        _Stream* stream
        void* outputbuffercpy = NULL
        void* inputbuffercpy = NULL

    stream = <_Stream*>(userData)

    # Returning 2 means aborting the stream immediately
    if status == RTAUDIO_INPUT_OVERFLOW:
        fprintf(stderr, 'Input overflow.\n')
        stream.stopThread.store(True)
        return 2
    elif status == RTAUDIO_OUTPUT_UNDERFLOW:
        fprintf(stderr, 'Output underflow.\n')
        # stream.stopThread.store(True)
        return 0

    if nFrames != stream.nFrames:
        printf('Number of frames mismath in callback data!\n')
        stream.stopThread.store(True)
        return 2

    if inputbuffer:
        # assert stream.inputQueue is not NULL
        inputbuffercpy = malloc(stream.inputbuffersize)
        memcpy(inputbuffercpy, inputbuffer,
               stream.inputbuffersize)
        stream.inputQueue.enqueue(inputbuffercpy)

    if outputbuffer:
        # assert stream.outputQueue is not NULL
        if stream.outputQueue.empty():
            fprintf(stderr, 'Stream output buffer underflow, zero-ing buffer...\n')
            # Pre-stack three empty output buffers
            # printf('Pre-stacking\n')
            #     outputbuffer = malloc(stream.outputbuffersize)
            memset(outputbuffer, 0, stream.outputbuffersize)
            if stream.inputQueue:
                stream.inputQueue.enqueue(NULL)
            return 0
        
        outputbuffercpy = stream.outputQueue.dequeue()
        memcpy(outputbuffer, outputbuffercpy,
               stream.outputbuffersize)
        free(outputbuffercpy)


    return 0


cdef void audioCallbackPythonThreadFunction(void* voidstream) nogil:
    cdef:
        _Stream* stream
        cnp.NPY_TYPES npy_format
        void* inputbuffer = NULL
        void* outputbuffer = NULL

    stream = <_Stream*> voidstream
    printf('Thread started\n')

    with gil:
        npy_format = _formats_rtkey[stream.sampleformat][2]
        callback = <object> stream.pyCallback
    while True:
        if stream.stopThread.load() == True:
            printf('Stopping thread...\n')
            return

        if stream.inputQueue:
            inputbuffer = stream.inputQueue.dequeue()
            # if inputbuffer == NULL:
            #     continue

        if stream.outputQueue:
            outputbuffer = malloc(stream.outputbuffersize)

        with gil:
            
            # Obtain stream information
            npy_input = None
            npy_output = None

            if stream.inputQueue and inputbuffer: 
                try:
                    npy_input = fromBufferToNPYNoCopy(
                            npy_format,
                            inputbuffer,
                            stream.inputParams.nChannels,
                            stream.nFrames)

                except Exception as e:
                    print('exception in cython callback for audio input: ', str(e))
                    return

            if stream.outputQueue: 
                try:
                    assert outputbuffer != NULL
                    npy_output = fromBufferToNPYNoCopy(
                            npy_format,
                            outputbuffer,
                            stream.outputParams.nChannels,
                            stream.nFrames)

                except Exception as e:
                    print('exception in Cython callback for audio output: ', str(e))
                    return
            try:
                rval = callback(npy_input,
                                npy_output,
                                stream.nFrames,
                               )

            except Exception as e:
                print('Exception in Cython callback: ', str(e))
                return

        if stream.outputQueue:
            stream.outputQueue.enqueue(outputbuffer)
            if not stream.inputQueue:
                while stream.outputQueue.size() > 10 and not stream.stopThread.load():
                    # printf('Sleeping...\n')
                    # No input queue to wait on, so we relax a bit here.
                    CPPsleep(1);

        # Outputbuffer is free'ed by the audiothread, so should not be touched
        # here.
        outputbuffer = NULL
        # Inputbuffer memory is owned by Numpy, so should not be free'ed
        inputbuffer = NULL


cdef void errorCallback(RtAudioError.Type _type,const string& errortxt) nogil:
    printf('RtAudio error callback called: ')
    printf(errortxt.c_str())
    printf('\n')


cdef class RtAudio:
    cdef:
        cppRtAudio _rtaudio
        _Stream* _stream

    def __cinit__(self):
        self._stream = NULL
        self._rtaudio.showWarnings(True)

    def __dealloc__(self):
        if self._stream is not NULL:
            fprintf(stderr, 'Force closing stream...')
            self._rtaudio.closeStream()
            self.cleanupStream(self._stream)

    cpdef unsigned int getDeviceCount(self):
        return self._rtaudio.getDeviceCount()

    cpdef unsigned int getDefaultOutputDevice(self):
        return self._rtaudio.getDefaultOutputDevice()

    cpdef unsigned int getDefaultInputDevice(self):
        return self._rtaudio.getDefaultInputDevice()

    def getDeviceInfo(self, unsigned int device):
        """
        Return device information of the current device
        """
        cdef cppRtAudio.DeviceInfo devinfo = self._rtaudio.getDeviceInfo(device)
        sampleformats = []
        nf = devinfo.nativeFormats
        for format_ in [ RTAUDIO_SINT8, RTAUDIO_SINT16, RTAUDIO_SINT24,
                RTAUDIO_SINT32, RTAUDIO_FLOAT32, RTAUDIO_FLOAT64]:
            if nf & format_:
                sampleformats.append(_formats_rtkey[format_][0])
        return DeviceInfo(
                index = device,
                probed = devinfo.probed,
                name = devinfo.name.decode('utf-8'),
                outputchannels = devinfo.outputChannels,
                inputchannels = devinfo.inputChannels,
                duplexchannels = devinfo.duplexChannels,
                samplerates = devinfo.sampleRates,
                sampleformats = sampleformats,
                prefsamplerate = devinfo.preferredSampleRate)

    @cython.nonecheck(True)
    def openStream(self,object outputParams,
                        object inputParams,
                        str sampleformat,
                        unsigned int sampleRate,
                        unsigned int bufferFrames,
                        object pyCallback,
                        object options = None,
                        object pyErrorCallback = None):
        """
        Opening a stream with specified parameters

        Args:
            outputParams: dictionary of stream outputParameters, set to None
            if no outputPararms are specified
            inputParams: dictionary of stream inputParameters, set to None
            if no inputPararms are specified
            sampleRate: desired sample rate.
            bufferFrames: the amount of frames in a callback buffer
            callback: callback to call. Note: this callback is called on a
            different thread!
            options: A dictionary of optional additional stream options
            errorCallback: client-defined function that will be invoked when an
            error has occured.

        Returns: None
        """
        if self._stream is not NULL:
            raise RuntimeError('Stream is already opened.')

        cdef:
            cppRtAudio.StreamParameters *rtOutputParams_ptr = NULL
            cppRtAudio.StreamParameters *rtInputParams_ptr = NULL
            cppRtAudio.StreamOptions streamoptions
            size_t bytespersample
            unsigned int bufferFrames_local

        streamoptions.flags = RTAUDIO_HOG_DEVICE
        streamoptions.numberOfBuffers = 4
        bufferFrames_local = bufferFrames

        self._stream = <_Stream*> malloc(sizeof(_Stream))
        if self._stream == NULL:
            raise MemoryError()

        self._stream.pyCallback = <PyObject*> pyCallback
        Py_INCREF(pyCallback)
        self._stream.sampleformat = _formats_strkey[sampleformat][0]
        self._stream.inputQueue = NULL
        self._stream.outputQueue = NULL
        self._stream.outputbuffersize = 0
        self._stream.inputbuffersize = 0
        self._stream.stopThread.store(False)
        self._stream.thread = NULL

        bytespersample = get_sampwidth_from_format_string(sampleformat)

        if outputParams is not None:
            rtOutputParams_ptr = &self._stream.outputParams
            rtOutputParams_ptr.deviceId = outputParams['deviceid']
            rtOutputParams_ptr.nChannels = outputParams['nchannels']
            rtOutputParams_ptr.firstChannel = outputParams['firstchannel']
            self._stream.outputQueue = new SafeQueue[void*]()

        if inputParams is not None:
            rtInputParams_ptr = &self._stream.inputParams
            rtInputParams_ptr.deviceId = inputParams['deviceid']
            rtInputParams_ptr.nChannels = inputParams['nchannels']
            rtInputParams_ptr.firstChannel = inputParams['firstchannel']
            self._stream.inputQueue = new SafeQueue[void*]()

        try:
            self._rtaudio.openStream(rtOutputParams_ptr,
                                     rtInputParams_ptr,
                                     _formats_strkey[sampleformat][0],
                                     sampleRate,
                                     &bufferFrames_local,
                                     audioCallback,
                                     <void*> self._stream,
                                     &streamoptions, # Stream options
                                     errorCallback  # Error callback
                                     ) 
            self._stream.nFrames = bufferFrames_local

        except Exception as e:
            print('Exception occured in stream opening: ', str(e))
            self.cleanupStream(self._stream)
            self._stream = NULL
            raise

        if inputParams is not None:
            self._stream.inputbuffersize = bufferFrames_local*bytespersample*inputParams['nchannels']
        if outputParams is not None:
            self._stream.outputbuffersize = bufferFrames_local*bytespersample*outputParams['nchannels']

        with nogil:
            self._stream.thread = new CPPThread[void*, void (*)(void*)](audioCallbackPythonThreadFunction,
                                                <void*> self._stream)
            CPPsleep(500)

        return bufferFrames_local

    cdef cleanupStream(self, _Stream* stream):
        # printf('Entrance function cleanupStream...\n')
        if stream == NULL:
            return

        with nogil:
            if stream.thread:
                stream.stopThread.store(True)
                if stream.inputQueue:
                    # If waiting in the input queue, hereby we let it run.
                    stream.inputQueue.enqueue(NULL)
                # printf('Joining thread...\n')
                # HERE WE SHOULD RELEASE THE GIL, as exiting the thread function
                # will require the GIL, which is locked by this thread!
                stream.thread.join()
                # printf('Thread joined!\n')
                del stream.thread

            if stream.outputQueue:
                del stream.outputQueue
            if stream.inputQueue:
                del stream.inputQueue

        if stream.pyCallback:
            Py_DECREF(<object> stream.pyCallback)

        free(stream)
        # printf('Cleanup of stream is done\n')

    def startStream(self):
        self._rtaudio.startStream()

    def stopStream(self):
        if self._stream is NULL:
            raise RuntimeError('Stream is not opened')
        try:
            self._rtaudio.stopStream()
        except:
            pass

    def closeStream(self):
        # print('closeStream')
        if self._stream is NULL:
            raise RuntimeError('Stream is not opened')
        # Closing stream
        self._rtaudio.closeStream()
        self.cleanupStream(self._stream)
        self._stream = NULL

    def abortStream(self):
        if self._stream is NULL:
            raise RuntimeError('Stream is not opened')
        self._rtaudio.abortStream()

    def isStreamOpen(self):
        return self._rtaudio.isStreamOpen()

    def isStreamRunning(self):
        return self._rtaudio.isStreamRunning()

    def getStreamTime(self):
        return self._rtaudio.getStreamTime()
    
    def setStreamTime(self, double time):
        return self._rtaudio.setStreamTime(time)

    
