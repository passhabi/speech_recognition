import numpy as np
# import warnings
import matplotlib.pyplot as plt
from librosa.display import specshow
from pydub import AudioSegment
import scipy
import os
import glob
from librosa import load
import pandas as pd

# For the project its more reliable to use the libraries that's out there instead of using our functions
#   that could go wrong and being distracted.

# importing the audio sounds:
# we will separate the data to the train and test set and for this I put my test set data to another folder to load.
sample_rate = 8000  # kilohertz , told by the sound provider.
signal_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: []
}
for wave_path in glob.glob("recordings/*.wav"):
    print(f"importing {wave_path}.")
    num = int(wave_path[11:12])  # take out the number of the sound wave from file name. recordings\8_theo_7.wav => 8
    # wave = AudioSegment.from_wav(wave_path)
    signal, _ = load(wave_path, sample_rate)

    signal_dict[num] += [signal]  # add an array column (the sound wave).

# Extracting MFCC Features. This library dose what we did in the exercises.
from librosa.feature import mfcc

mfcc_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: []
}

for key in signal_dict.keys():
    for i in range(len(signal_dict[key])):
        extracted_mfcc = mfcc(signal_dict[key][i], sample_rate, n_mfcc=12)
        mfcc_dict[key].append(extracted_mfcc.T)

## Mixture of gaussian.
# So in this section we know for each number we have MFCC feature vector. The next step is to understand this distribution of the data. In a simple word what probability each number have? We can use average or mean of the mfcc or we can use all the vector of MFCC to get a mixture of gaussian.
# For this we need mean vector MFCC and covariance matrix (note we wrote the cov matrix function in the exercises!)

# Create the model:
from hmmlearn import hmm

hmm_models = {}
for key in mfcc_dict.keys():
    model = hmm.GMMHMM(n_components=10)  # number of states, cause we have 10 digit.
    model.fit(np.vstack(mfcc_dict[key]))
    hmm_models[key] = model

# Testing The model
# In this section we should measure how good is our model. For this We need to do the same in the test data and find out which of 10 models that we created has max probability on the given signal and for that signal the corresponding model with digit is our prediction.
import pandas as pd
test_df = pd.DataFrame()
for wave_path in glob.glob(
        "test_recordings/*.wav"):  # note any of the data in this folder is not seeing by the model in the training.
    print(f"importing {wave_path}.")
    num = int(wave_path[16:17])  # take out the number of the sound wave from file name.
    signal, _ = load(wave_path, sample_rate)

    temp = pd.DataFrame(data=[[signal, num]], columns=['signal', 'digit'])
    test_df = test_df.append(temp, ignore_index=True)


# As before we don't need the signals we just need features of the signal and for this due, we use MFCC again.
# Let's apply MFCC feature on the test set:
def mffc_extraction(signal):
    return mfcc(signal, sample_rate, n_mfcc=12).T


test_df.signal = test_df.signal.apply(mffc_extraction)
# Since I replaced signals with mfcc I should rename the column's name:
test_df.rename(columns={"signal": "mfcc"}, inplace=True)

prediction = []
for i in test_df.mfcc:
    # for each data in test set do:
    prob = np.array(-np.inf)  # todo: minus one index is the right label.
    for num, model in zip(hmm_models.keys(), hmm_models.values()):  # see which of the models has most score:
        probability = model.score(i)
        prob = np.append(prob, probability)
    prediction += [np.argmax(
        prob) - 1]  # the corresponding index is the predicted digit. 'predictions' list corresponds with the test data. (but -1)

test_df["predicted"] = prediction  # save to the test data frame.

num_correct = 0
for i, j in zip(test_df.digit, test_df.predicted):
    if int(i) == int(j):
        num_correct += 1

n = len(test_df)
accuracy = num_correct * (100 / n)
print("accuracy:", accuracy)

