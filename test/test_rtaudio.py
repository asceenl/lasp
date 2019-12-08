#!/usr/bin/python3
import numpy as np
from lasp_rtaudio import RtAudio, SampleFormat, Format_SINT32, Format_FLOAT64
import time


nframes = 0
samplerate = 48000
omg = 2*np.pi*1000

def mycallback(input_, nframes, streamtime):
    t = np.linspace(streamtime, streamtime + nframes/samplerate,
            nframes)[np.newaxis,:]
    outp = 0.1*np.sin(omg*t)
    return outp, 0

if __name__ == '__main__':
    pa = RtAudio()
    count = pa.getDeviceCount()
    # dev = pa.getDeviceInfo(0)
    for i in range(count):
        dev = pa.getDeviceInfo(i)
        print(dev)
   
    outputparams = {'deviceid': 0, 'nchannels': 1, 'firstchannel': 0}
    pa.openStream(outputparams, None , Format_FLOAT64,samplerate, 512, mycallback)
    pa.startStream()

    input()

    pa.stopStream()
    pa.closeStream()



