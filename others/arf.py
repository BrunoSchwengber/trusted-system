import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
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

# Drift Detector
# S: Source (Old Data)
# T: Target (New Data)
# ST: S&T combined
def drift_detector(S,T,threshold = 0.70):
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
	#clf = LogisticRegression(solver='liblinear')
	#clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
	clf = DecisionTreeClassifier(random_state=0)
	#clf = GaussianNB()
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
	recall = recall_score(labels, predictions)
	precision = precision_score(labels, predictions)
	print(recall)
	# Signal drift if AUC is larger than the threshold

	if auc_score > threshold:
		return True
	else:
		return False


class D3():
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
		if drift_detector(self.win_data[:self.w], self.win_data[self.w:self.size], auc): #returns true if drift is detected
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
		return self.win_data[:self.window_index]
	def getCurrentLabels(self):
		return self.win_label[:self.window_index]


def select_data(x):
	df = pd.read_csv(x, header=0)
	df = shuffle(df)
	df = df.drop(columns=['IP_Origem'])
	df = df.drop(columns=['IP_Destino'])
	scaler = MinMaxScaler()
	df.iloc[:,0:df.shape[1]-1] = scaler.fit_transform(df.iloc[:,0:df.shape[1]-1])
	return df


def check_true(y,y_hat):
	if(y==y_hat):
		return 1
	else:
		return 0



df = select_data(sys.argv[1])
dataset = str(sys.argv[1])
label = pd.read_csv(sys.argv[5])
stream = DataStream(df, y=label)
#stream.prepare_for_use()


#stream_clf = HoeffdingTree() #HAT()
stream_clf = AdaptiveRandomForestClassifier()
#adwin = ADWIN()

w = int(sys.argv[2])
rho = float(sys.argv[3])
auc = float(sys.argv[4])

stream_acc = []
stream_pre = []
stream_recall = []
stream_fscore = []
stream_record = []
stream_true= 0

j=0
k=0
i=0
drifts=[]
start = time.time()
X,y = stream.next_sample(int(w*rho))
#print(stream.target_values)
stream_clf.partial_fit(X, y)#, classes=stream.target_values)
#print(y)
y_true=[]
y_pred=[]
while(stream.has_more_samples()):
	i=i+1

	X,y = stream.next_sample()

	y_hat = stream_clf.predict(X)
	stream_true = stream_true + check_true(y, y_hat)
	y_true.append(y)
	y_pred.append(y_hat)
	stream_clf.partial_fit(X,y)
	stream_acc.append(stream_true / (i+1))
	#adwin.add_element(check_true(y, y_hat))
	#if adwin.detected_change():
	#	drifts.append(i)
	stream_pre.append(precision_score(y_true, y_pred, average='weighted', zero_division=1))
	stream_recall.append(recall_score(y_true, y_pred, average='weighted', zero_division=1))
	stream_fscore.append(f1_score(y_true, y_pred, average='weighted', zero_division=1))

	stream_record.append(check_true(y, y_hat))
	i = i+1

elapsed = format(time.time() - start, '.4f')
acc = format((stream_acc[-1]*100), '.4f')
final_accuracy = "Final accuracy: {}, Elapsed time: {}".format(acc,elapsed)
print(final_accuracy)
print(drifts)

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

#min_max_scaler = preprocessing.MinMaxScaler()
#drifts = min_max_scaler.fit_transform(drifts)


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

#drifts = normalize(drifts)


a=int(len(df)/30)
ddd_acc2 = window_average(stream_record, a)
ddd_pre2 = window_average(stream_pre, a)
ddd_rec2 = window_average(stream_recall, a)
ddd_fsc2 = window_average(stream_fscore, a)
#print(k)
#print(j)


x = np.linspace(0, 100, len(ddd_acc2), endpoint=True)

arquivo =open("arf_{}.csv".format(dataset), 'a')
for i, j, k, l in zip(ddd_acc2, ddd_pre2, ddd_rec2, ddd_fsc2):
    string = str(i)+','+str(j)+','+str(k)+','+str(l)+','+'\n'
    arquivo.write(string)
arquivo.close()

f = plt.figure()
plt.plot(x, ddd_acc2, 'r', label='ARF', marker="*") 

plt.xlabel('Percent of data', fontsize=30)
plt.ylabel('Accuracy', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.grid(True)
#plt.hlines(final_accuracy/100, linestyles='dashed', linewidth=0.1)
plt.legend(loc='lower left', fontsize=30)
plt.ylim(0,)
for i in drifts:
	x = i*100
	plt.vlines(x,ymin=0.3, ymax=1, linestyles='dashed', linewidth=0.1)
	
print(classification_report(y_pred, y_true, target_names=['class 0', 'class 1']))
#plt.show()

#f.savefig("d3-perceptron-recall.png", bbox_inches='tight')

'''eval = EvaluatePrequential(
							pretrain_size=1000,
							output_file='result_',
							batch_size=1,
							metrics=['accuracy'],
							#data_points_for_classification=True,
							n_wait=500,
							max_time=1000000000,
							show_plot=True)
	# 4. Run
eval.evaluate(stream=stream, model=stream_clf, model_names=['HT', 'HAT'])
'''