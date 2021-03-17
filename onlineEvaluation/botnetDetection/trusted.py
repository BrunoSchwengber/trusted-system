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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class Trusted:
    def __init__(self, distancia, wo, rhoo, auco):
        DISTANCIA= float(distancia)
        SIMILARIDADE=(DISTANCIA*0.005)

        self.stream_clf = AGLO(DISTANCIA, SIMILARIDADE)

        w = int(wo)
        rho = float(rhoo)
        auc = float(auco)

        self.D3_win = D3(w,rho,8,auc)#stream.n_features,auc)
        stream_acc = []
        stream_record = []
        stream_true= 0

        i=0
        j=0
        k=0
        y_true=[]
        y_pred = []
        acuracia=[]
        precisao=[]
        recall=[]
        fscore=[]

        drifts=[]
        #start = time.time()
        #X,y = stream.next_sample(int(w*rho))
        #for i, a in zip(X,y):
        #    D3_win.addInstance(i,a)

        #stream_clf.fit(X, y)

    def normalize(self, x):
        """
            argument
                - x: input image data in numpy array [32, 32, 3]
            return
                - normalized x 
        """
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x-min_val) / (max_val-min_val)
        return x

    def window_average(self, x,N):
        low_index = 0
        high_index = low_index + N
        w_avg = []
        while(high_index<len(x)):
            temp = sum(x[low_index:high_index])/N
            w_avg.append(temp)
            low_index = low_index + N
            high_index = high_index + N
        return w_avg

    def select_data(self, x):
        df = pd.read_csv(x, header=0)
        df = df.drop(columns=['IP_Origem'])
        df = df.drop(columns=['IP_Destino'])
        scaler = MinMaxScaler()
        df.iloc[:,0:df.shape[1]-1] = scaler.fit_transform(df.iloc[:,0:df.shape[1]-1])
        return df

    def check_true(y, y_hat):
        return np.array_equal(y, y_hat)

    def test (self, data, labels):
        y_hat = self.stream_clf.hac(data)
        self.acuracia.append(accuracy_score(labels, y_hat))
        self.precisao.append(precision_score(labels, y_hat, average='weighted', zero_division=1))
        self.recall.append(recall_score(labels, y_hat, average='weighted', zero_division=1))
        self.fscore.append(f1_score(labels, y_hat, average='weighted', zero_division=1))

        #return acuracia, precisao, recall, fscore

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
        self.acuracia=[]
        self.precisao=[]
        self.recall=[]
        self.fscore=[]
        while(stream.has_more_samples()):            

            X,y = stream.next_sample()
            if self.D3_win.isEmpty():
                
                self.D3_win.addInstance(X,y)
                self.test(self.D3_win.getCurrentData(), self.D3_win.getCurrentLabels())

            else:
                
                if self.D3_win.driftCheck():             #detected
                    drifts.append(i)
                    print("concept drift detected at {}".format(i))
                    self.test(self.D3_win.getCurrentData(), self.D3_win.getCurrentLabels())
                    self.D3_win.addInstance(X,y)
                
                else:
                    k=k+1
                    self.test(self.D3_win.getCurrentData(), self.D3_win.getCurrentLabels())
                    self.D3_win.addInstance(X,y)

        return self.acuracia, self.precisao, self.recall, self.fscore      

