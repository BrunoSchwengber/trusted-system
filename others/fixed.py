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


df = select_data(sys.argv[1])
dataset = str(sys.argv[1])
w = int(sys.argv[2])
label = pd.read_csv(sys.argv[3])
stream = DataStream(df, y=label)
#stream.prepare_for_use()

DISTANCIA=0.1
SIMILARIDADE=0.0005
stream_clf = AGLO(DISTANCIA, SIMILARIDADE)

acuracia=[]
precisao=[]
recall=[]
fscore=[]

start = time.time()

while(stream.has_more_samples()):

    X,y = stream.next_sample(w)
        
    y_hat = stream_clf.hac(X)
    
    acuracia.append(accuracy_score(y, y_hat))
    precisao.append(precision_score(y, y_hat, average='weighted', zero_division=1))
    recall.append(recall_score(y, y_hat, average='weighted', zero_division=1))
    fscore.append(f1_score(y, y_hat, average='weighted', zero_division=1))

elapsed = format(time.time() - start, '.4f')
final_accuracy = "Final accuracy: {}, precision: {}, recall:{}, fscore:{}, Elapsed time: {}".format(mean(acuracia), mean(precisao), mean(recall), mean(fscore), elapsed)
print("Delta: {}, DIST: {}, Sim: {}".format(w, DISTANCIA, SIMILARIDADE))
print(final_accuracy)
print(len(acuracia))

a=int(len(acuracia)/30)


ddd_acc2 = window_average(acuracia, a)
ddd_pre2 = window_average(precisao, a)
ddd_rec2 = window_average(recall, a)
ddd_fsc2 = window_average(fscore, a)

#print(ddd_acc2)
x = np.linspace(0, 100, len(ddd_acc2), endpoint=True)

f = plt.figure()
plt.plot(x, ddd_acc2, 'r', label='Fixed HAC', marker="*") 

plt.xlabel('Percent of data', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)
plt.ylim(0,1)
plt.grid(True)
plt.legend(loc='lower left', fontsize=24)



arquivo =open("fixedAglo_{}_{}.csv".format(dataset, w), 'a')
for i, j, k, l in zip(ddd_acc2, ddd_pre2, ddd_rec2, ddd_fsc2):
    string = str(i)+','+str(j)+','+str(k)+','+str(l)+','+'\n'
    arquivo.write(string)
arquivo.close()

f.savefig("FIXED w={}_{}.png".format(w, dataset), bbox_inches='tight')