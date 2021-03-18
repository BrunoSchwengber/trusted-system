#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.cluster import AgglomerativeClustering
import silhueta
import pandas as pd
import numpy as np
import numpy_indexed as npi
from sklearn import preprocessing
from statistics import mean
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn import metrics
import sys

class HAC():

	def __init__(self, dist, thresh):
		self.DISTANCIA = dist
		self.THRESHOLD = thresh
		self.clusterer = AgglomerativeClustering(n_clusters=None,
											distance_threshold=self.DISTANCIA,
											linkage='average',
											affinity='euclidean') #affinity='manhattan')

	def fit(self, X, y):
		data = pd.DataFrame(X)
		min_max_scaler = preprocessing.MinMaxScaler()
		data = min_max_scaler.fit_transform(data)
		data = pd.DataFrame(data)
		self.clusterer.fit(X)

	def hac(self,X):
		#print(X)
		data = pd.DataFrame(X)
		min_max_scaler = preprocessing.MinMaxScaler()
		data = min_max_scaler.fit_transform(data)
		data = pd.DataFrame(data)
		
		data = data.rename_axis(' ').values
		
		self.clusterer.fit_predict(data)

		#print("Calculando Similaridades")
		intra_cluster_dists = silhueta.silhouette_samples(data, self.clusterer.labels_, metric='euclidean')# metric='manhattan')
		mean_dist_cluster = npi.group_by(self.clusterer.labels_).mean(intra_cluster_dists)

		similarity= list(mean_dist_cluster)

		similarity[1] = np.array(similarity[1])

		similarity[1] = list(similarity[1])

		valores_similaridade = similarity[1]

		indexes=[]

		for i in range(len(valores_similaridade)):
			if valores_similaridade[i] <= self.THRESHOLD:
				indexes.append(valores_similaridade.index(valores_similaridade[i])-1)

		indexes2 = []

		#print("Adicionando Rotulos")
		for i in range(len(self.clusterer.labels_)):
			if (self.clusterer.labels_[i] in indexes) and (self.clusterer.labels_[i] not in indexes2): 
				self.clusterer.labels_[i] = -2
		for i in range(len(self.clusterer.labels_)):
			if self.clusterer.labels_[i] >= -1:
				self.clusterer.labels_[i] = 0
		for i in range(len(self.clusterer.labels_)):
			if self.clusterer.labels_[i] == -2:
				self.clusterer.labels_[i]=1
		
		return self.clusterer.labels_
