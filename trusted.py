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

def normalize(x):
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

def select_data(x):
    df = pd.read_csv(x, header=0)
    df = df.drop(columns=['IP_Origem'])
    df = df.drop(columns=['IP_Destino'])
    scaler = MinMaxScaler()
    df.iloc[:,0:df.shape[1]-1] = scaler.fit_transform(df.iloc[:,0:df.shape[1]-1])
    return df

def check_true(y, y_hat):
    return np.array_equal(y, y_hat)

df = select_data(sys.argv[1])
dataset = str(sys.argv[1])
label = pd.read_csv(sys.argv[5])
stream = DataStream(df, y=label)
#stream.prepare_for_use()

distance= float(sys.argv[6])

SIMILARIDADE=(distance*0.005)

stream_clf = AGLO(distance, SIMILARIDADE)

w = int(sys.argv[2])
rho = float(sys.argv[3])
auc = float(sys.argv[4])

D3_win = D3(w,rho,stream.n_features,auc)
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
start = time.time()
X,y = stream.next_sample(int(w*rho))
for i, a in zip(X,y):
    D3_win.addInstance(i,a)

stream_clf.fit(X, y)

while(stream.has_more_samples()):
    

    X,y = stream.next_sample()
    if D3_win.isEmpty():
        
        D3_win.addInstance(X,y)
        y_hat = stream_clf.hac(D3_win.getCurrentData())
       
        acuracia.append(accuracy_score(D3_win.getCurrentLabels(), y_hat))
        precisao.append(precision_score(D3_win.getCurrentLabels(), y_hat, average='weighted', zero_division=1))
        recall.append(recall_score(D3_win.getCurrentLabels(), y_hat, average='weighted', zero_division=1))
        fscore.append(f1_score(D3_win.getCurrentLabels(), y_hat, average='weighted', zero_division=1))

    else:
        if D3_win.driftCheck():             #detected
            #print("drift detectado")
            j=j+1
            drifts.append(i)
            #print("concept drift detected at {}".format(i))
            #retrain the model
            
            y_hat = stream_clf.hac(D3_win.getCurrentData())
            
            #evaluate and update the model
            acuracia.append(accuracy_score(D3_win.getCurrentLabels(), y_hat))
            precisao.append(precision_score(D3_win.getCurrentLabels(), y_hat, average='weighted', zero_division=1))
            recall.append(recall_score(D3_win.getCurrentLabels(), y_hat, average='weighted', zero_division=1))
            fscore.append(f1_score(D3_win.getCurrentLabels(), y_hat, average='weighted', zero_division=1))
            
            D3_win.addInstance(X,y)
        else:
            #print("no drift")
            k=k+1
            #evaluate and update the model
            y_hat = stream_clf.hac(D3_win.getCurrentData())
            
            acuracia.append(accuracy_score(D3_win.getCurrentLabels(), y_hat))
            precisao.append(precision_score(D3_win.getCurrentLabels(), y_hat, average='weighted', zero_division=1))
            recall.append(recall_score(D3_win.getCurrentLabels(), y_hat, average='weighted', zero_division=1))
            fscore.append(f1_score(D3_win.getCurrentLabels(), y_hat, average='weighted', zero_division=1))

            #add new sample to the window
            D3_win.addInstance(X,y)
    i = i+1  


a=int(len(df)/30)
ddd_acc2 = window_average(acuracia, a)
ddd_pre2 = window_average(precisao, a)
ddd_rec2 = window_average(recall, a)
ddd_fsc2 = window_average(fscore, a)


elapsed = format(time.time() - start, '.4f')

final_accuracy = "Final accuracy: {}, precision: {}, recall:{}, f-score:{}, Elapsed time: {}".format(mean(acuracia), mean(precisao), mean(recall), mean(fscore), elapsed)
print("Delta: {}, Sigma: {}, Treshold: {}, Distance: {}, Similarity: {}".format(w, rho, auc, distance, SIMILARIDADE))
print(final_accuracy)
#print(drifts)

a=int(len(acuracia)/30)
ddd_acc2 = window_average(acuracia, a)
#print(k)
#print(j)

drifts = normalize(drifts)
x = np.linspace(0, 100, len(ddd_acc2), endpoint=True)

f = plt.figure()
plt.plot(x, ddd_acc2, 'r', label='TRUSTED', marker="*") 

plt.xlabel('Percent of data', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)
plt.ylim(0,1)
plt.grid(True)
plt.legend(loc='lower left', fontsize=24)
plt.ylim(0,)

for i in drifts:
    x = i*100
    plt.vlines(x,ymin=0.3, ymax=1, linestyles='dashed', linewidth=0.1)


arquivo =open("trusted_{}_{}_{}_{}_{}.csv".format(dataset, w, rho, auc, distance), 'a')
for i, j, k, l in zip(ddd_acc2, ddd_pre2, ddd_rec2, ddd_fsc2):
    string = str(i)+','+str(j)+','+str(k)+','+str(l)+','+'\n'
    arquivo.write(string)
arquivo.close()

plt.show()

f.savefig("trusted_{}_{}_{}_{}_{}.png".format(dataset,w, rho, auc, distance), bbox_inches='tight')