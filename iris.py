#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:26:08 2024

@author: filip
"""
import numpy as np
import random
from sklearn.datasets import load_iris
import pandas as pd

def iris_dataset_generator(epoch_size):
    iris=load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    
    X = iris.data
    Y = iris.target
    
    permutation = list(np.random.permutation(150))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    
    mini_batches = []
    
    for i in range(0,5):
        mini_batch_X = shuffled_X[i:(i+1)*epoch_size]
        mini_batch_Y = shuffled_Y[i:(i+1)*epoch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches

def average_value_generator(mini_batch_X, i, indices):
    
    mini_batch_X, mini_batch_Y = mini_batch
    indices = np.wherenp.where(mini_batch_Y == i)[0]
    samples = mini_batch_X[indices, :]
    average_X = np.mean(samples, axis=0)
    return average_X

def target_value_generator(mini_batch_Y, Y, i, indices):
    
    mini_batch_X, mini_batch_Y = mini_batch
    indices = np.wherenp.where(mini_batch_Y == i)[0]
    samples = mini_batch_X[indices, :]
    average_Y = np.mean(samples, axis=0)
    return average_X 
    
if __name__ == "__main__":
    epoch_size=10
    iris_dataset_generator(epoch_size)