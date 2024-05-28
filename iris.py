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



#need to add two additional columns
def iris_dataset_generator(epoch_size):
    iris = load_iris()
    X = iris.data
    Y = iris.target
    
    
    permutation = list(np.random.permutation(len(X)))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    
    mini_batches = []
    
    num_batches = len(X) // epoch_size
    
    for i in range(num_batches):
        mini_batch_X = shuffled_X[i * epoch_size:(i + 1) * epoch_size]
        mini_batch_Y = shuffled_Y[i * epoch_size:(i + 1) * epoch_size] + 1
        mini_batch_Y = np.column_stack((mini_batch_Y.reshape(-1, 1), np.zeros((epoch_size, 2))))
        mini_batch = (mini_batch_X, mini_batch_Y) 
        mini_batches.append(mini_batch)
    
    # Handle the last mini-batch which might be smaller than epoch_size
    if len(X) % epoch_size != 0:
        mini_batch_X = shuffled_X[num_batches * epoch_size:]
        mini_batch_Y = shuffled_Y[num_batches * epoch_size:]
        mini_batch_Y = np.column_stack((mini_batch_Y.reshape(-1, 1), np.zeros((len(mini_batch_Y), 2))))
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    print(f"Generated {len(mini_batches)} mini-batches")  # Debug: Check number of mini-batches generated
    
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
    
# if __name__ == "__main__":
#     epoch_size=10
#     iris_dataset_generator(epoch_size)