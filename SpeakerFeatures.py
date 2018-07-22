# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 01:38:06 2018

@author: RızaIşık
"""
import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

def delta(arr):
    rows,cols = arr.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = (arr[index[0][0]]-arr[index[0][1]] + (2 * (arr[index[1][0]]-arr[index[1][1]]))) / 10
    return deltas

def getMFCC(audio,rate):   
    feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy = True)
    feature = preprocessing.scale(feature)
    combined = np.hstack((feature, delta(feature))) 
    return combined
    
if __name__ == "__main__":
     print ('In main, Call extract_features(audio,signal_rate) as parameters')