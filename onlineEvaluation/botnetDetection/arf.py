import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.data.data_stream import DataStream
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.bayes.naive_bayes import NaiveBayes
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from sklearn import preprocessing
import time
import sys
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (9,7)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 30

class Arf:
	def __init__(self):
		self.stream_clf = AdaptiveRandomForestClassifier()

	def check_true(y,y_hat):
		if(y==y_hat):
			return 1
		else:
			return 0

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
		y_true = []
		y_pred = []
		while(stream.has_more_samples()):
			
			X,y = stream.next_sample()

			y_hat = self.stream_clf.predict(X)
			
			y_true.append(y)
			y_pred.append(y_hat)
			self.stream_clf.partial_fit(X,y)
			
			acuracia.append(accuracy_score(y_true, y_pred))
			precisao.append(precision_score(y_true, y_pred, average='weighted', zero_division=1))
			recall.append(recall_score(y_true, y_pred, average='weighted', zero_division=1))
			fscore.append(f1_score(y_true, y_pred, average='weighted', zero_division=1))

		return acuracia, precisao, recall, fscore


	def select_data(x):
		df = pd.read_csv(x, header=0)
		df = shuffle(df)
		df = df.drop(columns=['IP_Origem'])
		df = df.drop(columns=['IP_Destino'])
		scaler = MinMaxScaler()
		df.iloc[:,0:df.shape[1]-1] = scaler.fit_transform(df.iloc[:,0:df.shape[1]-1])
		return df

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
