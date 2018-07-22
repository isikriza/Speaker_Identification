# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 01:39:28 2018

@author: RızaIşık
"""
import os
import time
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GMM 
from SpeakerFeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
   
dest_speaker = "speaker_models\\" 
dest_gender = "gender_models\\"

########################## SPEAKER TRAINING ###################################
source_speaker = "train\\speaker_train\\"   

file_paths = [os.path.join(source_speaker,fname) for fname in
              os.listdir(source_speaker) if fname.endswith('.wav')]

for path in file_paths:
    print (path)
    sr,audio = read(path)
    vector = extract_features(audio,sr)
    gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)
    gmm.fit(vector)
        
    picklefile = path.split(".wav")[0].split("\\")[2] + ".gmm"
    cPickle.dump(gmm,open(dest_speaker + picklefile,'wb'))
    print ('+ modeling completed for speaker:', picklefile, " with data point = ", vector.shape)
    time.sleep(1.0)
    
########################## MALE TRAINING ######################################
source_male = "train\\male_train\\"

file_paths = [os.path.join(source_male,fname) for fname in
              os.listdir(source_male) if fname.endswith('.wav')]

features = np.asarray(());
for path in file_paths:
    sr,audio = read(path)
    vector = extract_features(audio,sr)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

gmm = GMM(n_components = 8, n_iter = 200, covariance_type='diag', n_init = 3)
gmm.fit(features)

picklefile = "Male.gmm"
cPickle.dump(gmm,open(dest_gender + picklefile,'wb'))

########################## FEMALE TRAINING ####################################
source_female = "train\\female_train\\"

file_paths = [os.path.join(source_female,fname) for fname in
              os.listdir(source_female) if fname.endswith('.wav')]

features = np.asarray(());
for path in file_paths:
    sr,audio = read(path)
    vector = extract_features(audio,sr)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

gmm = GMM(n_components = 8, n_iter = 200, covariance_type='diag', n_init = 3)
gmm.fit(features)

picklefile = "Female.gmm"
cPickle.dump(gmm,open(dest_gender + picklefile,'wb'))