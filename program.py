from pydub import AudioSegment
# from pydub.utils import mediainfo
from pydub import playback, utils
import python_speech_features as psf
import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt

# importing the audio sounds:
previous_num = 0  # storing the number of the voice sound. It stats at 0 till 9.
signal_arrays = [
    []  # by default has a row. index row of sound waves corresponding to the number.
]  # to storing the sound waves.

for wave_path in glob.glob("recordings/*.wav"):
    print(f"importing {wave_path}.")
    wave = AudioSegment.from_wav(wave_path)
    num = int(wave_path[11:12])  # take out the number of the sound wave from file name. recordings\8_theo_7.wav => 8

    if previous_num != num:  # <if the number has increased>:
        signal_arrays.append([])  # add a new row.

    signal = np.fromstring(wave.raw_data, 'int16')  # convert wave file to numpy array.
    signal_arrays[num].append(signal)  # add an array column (the sound wave).

    previous_num = num  # update number index

# playback.play(signal_arrays[3][50])

# framing the signals:
# todo: its just for a signal:
sig = signal_arrays[0][0]

sample_rate = 8000  # the audio files are 8khz

frame_size = round(0.025 * sample_rate)
overlap_size = round(0.010 * sample_rate)

frame_starts = np.arange(0, len(sig), frame_size - overlap_size)  # frame_end(i) =  frame_start(i) + frame_size

# extract MFCC feature of the signal audio
#   1. calculate DFT (Discrete Fourier Transform) of the frames: si(n) denote the nth sample in the ith frame.
num_zero_pad = (frame_starts[-1] + frame_size) \
               - len(sig)  # If the speech file does not divide into an even number of frames, pad it with zeros
sig = np.append(sig, [0] * num_zero_pad)
fft = list()
for start in frame_starts:  # <for each sample in the frame>:
    end = start + frame_size
    fft.append(abs(np.fft.fft(sig[start:end])))  # take the Discrete Fourier Transform of the frame.

plt.plot(fft[30])
plt.show()
