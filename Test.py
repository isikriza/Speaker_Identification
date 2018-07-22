# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 01:38:50 2018

@author: RızaIşık
"""
import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from SpeakerFeatures import getMFCC
import warnings
warnings.filterwarnings("ignore")
import time

source = "test\\"   

modelPath = "speaker_models\\"

genderModelPath = "gender_models\\"   

file_paths = [os.path.join(source,fname) for fname in
              os.listdir(source) if fname.endswith('.wav')]

gmm_files = [os.path.join(modelPath,fname) for fname in 
              os.listdir(modelPath) if fname.endswith('.gmm')]

gmm_files_gender = [os.path.join(genderModelPath,fname) for fname in 
              os.listdir(genderModelPath) if fname.endswith('.gmm')]

models = [cPickle.load(open(fname,'rb')) for fname in gmm_files]

genderModels = [cPickle.load(open(fname,'rb')) for fname in gmm_files_gender]

speakers = [fname.split(".gmm")[0].split("\\")[1] for fname 
              in gmm_files]

genders = [fname.split(".gmm")[0].split("\\")[1] for fname 
          in gmm_files_gender]
 
for path in file_paths:   
    print (path)
    sr,audio = read(path)
    vector = getMFCC(audio,sr)
    
    log_likelihood_speaker = np.zeros(len(models))
    log_likelihood_gender = np.zeros(len(genderModels))
    
    for i in range(len(models)):
        gmm_speaker = models[i]
        
        scores = np.array(gmm_speaker.score(vector))
        log_likelihood_speaker[i] = scores.sum()
    
    for i in range(len(genderModels)):
        gmm_gender = genderModels[i]
        
        scores = np.array(gmm_gender.score(vector))
        log_likelihood_gender[i] = scores.sum()
        
    winner_speaker = np.argmax(log_likelihood_speaker)
    winner_gender = np.argmax(log_likelihood_gender)
    
    print ("\tdetected as - ", speakers[winner_speaker],
           " Gender: ", genders[winner_gender])
    time.sleep(1.0)