#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import roc_auc_score as AUC
from sklearn import preprocessing
from sklearn.utils import shuffle
from skmultiflow.data.data_stream import DataStream
from sklearn.tree import DecisionTreeClassifier


# Drift Detector
# S: Source (Old Data)
# T: Target (New Data)
# ST: S&T combined
def drift_detector(S,T,threshold):
    T = pd.DataFrame(T)
    S = pd.DataFrame(S)
    # Give slack variable in_target which is 1 for old and 0 for new
    T['in_target'] = 0 # in target set
    S['in_target'] = 1 # in source set
    # Combine source and target with new slack variable 
    ST = pd.concat( [T, S], ignore_index=True, axis=0)
    labels = ST['in_target'].values
    ST = ST.drop('in_target', axis=1).values
    # You can use any classifier for this step. We advise it to be a simple one as we want to see whether source
    # and target differ not to classify them.
    clf = DecisionTreeClassifier(random_state=0, max_depth=4)
    predictions = np.zeros(labels.shape)

    # Divide ST into two equal chunks
    # Train LR on a chunk and classify the other chunk
    # Calculate AUC for original labels (in_target) and predicted ones
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_idx, test_idx in skf.split(ST, labels):
        X_train, X_test = ST[train_idx], ST[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict(X_test)#[:, 1]
        predictions[test_idx] = probs
    auc_score = AUC(labels, predictions)
    #recall = recall_score(labels, predictions)
    #precision = precision_score(labels, predictions)
    # Signal drift if AUC is larger than the threshold

    if auc_score > threshold:
        return True
    else:
        return False


class D3N():
    def __init__(self, w, rho, dim, auc):
        self.size = int(w*(1+rho))
        self.win_data = np.zeros((self.size,dim))
        self.win_label = np.zeros(self.size)
        self.w = w
        self.rho = rho
        self.dim = dim
        self.auc = auc
        self.drift_count = 0
        self.window_index = 0
    def addInstance(self,X,y):
        if(self.isEmpty()):
            self.win_data[self.window_index] = X
            self.win_label[self.window_index] = y
            self.window_index = self.window_index + 1
        else:
            print("Error: Buffer is full!")
    def isEmpty(self):
        return self.window_index < self.size
    def driftCheck(self):
        if drift_detector(self.win_data[:self.w], self.win_data[self.w:self.size], self.auc): #returns true if drift is detected
            self.window_index = int(self.w * self.rho)
            self.win_data = np.roll(self.win_data, -1*self.w, axis=0)
            self.win_label = np.roll(self.win_label, -1*self.w, axis=0)
            self.drift_count = self.drift_count + 1
            return True
        else:
            self.window_index = self.w
            self.win_data = np.roll(self.win_data, -1*(int(self.w*self.rho)), axis=0)
            self.win_label =np.roll(self.win_label, -1*(int(self.w*self.rho)), axis=0)
            return False
    def getCurrentData(self):
        return self.win_data #[:self.window_index]
    def getCurrentLabels(self):
        return self.win_label #[:self.window_index]