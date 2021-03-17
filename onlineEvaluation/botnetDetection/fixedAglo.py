#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from D3 import D3
from statistics import mean
from aglomerative import AGLO
from sklearn.preprocessing import MinMaxScaler
from skmultiflow.data.data_stream import DataStream
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix, classification_report

class Fixed:
    def __init__(self, distancia):
        DISTANCIA=float(distancia)
        SIMILARIDADE=(DISTANCIA*0.005)

        self.stream_clf = AGLO(DISTANCIA, SIMILARIDADE)

        acuracia=[]
        precisao=[]
        recall=[]
        fscore=[]

    def select_data(x):
        df = pd.read_csv(x, header=0)
        df = df.drop(columns=['IP_Origem'])
        df = df.drop(columns=['IP_Destino'])
        scaler = MinMaxScaler()
        df.iloc[:,0:df.shape[1]-1] = scaler.fit_transform(df.iloc[:,0:df.shape[1]-1])
        return df

    def check_true(y, y_hat):
        return np.array_equal(y, y_hat)

    def window_average(x,N):
        low_index = 0
        high_index = low_index + N
        w_avg = []
        while(high_index<len(x)):
            temp = sum(x[low_index:high_index])/N
            w_avg.append(temp)
            low_index = low_index + N
            high_index = high_index + N
        return w_avg

    def process(self, dados):
        """
            argument
                - dados: input dataframe for classification
            return
                - acuracia, precisao, recall e fscore 
        """
        df = dados.drop(columns=['src', 'dst'])
        label = df.iloc[:,8]
        label = pd.DataFrame(label)
        #print(label)
        df = df.drop(columns=['label'])

        stream = DataStream(df, y=label)
        acuracia=[]
        precisao=[]
        recall=[]
        fscore=[]
        cont=0
        while(stream.has_more_samples()):
            
            X,y = stream.next_sample(int(stream.n_remaining_samples()))
                
            y_hat = self.stream_clf.hac(X)
            
            acuracia.append(accuracy_score(y, y_hat))
            precisao.append(precision_score(y, y_hat, average='weighted', zero_division=1))
            recall.append(recall_score(y, y_hat, average='weighted', zero_division=1))
            fscore.append(f1_score(y, y_hat, average='weighted', zero_division=1))

        return acuracia, precisao, recall, fscore    