import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from cvxopt import matrix, solvers
from sklearn.ensemble import RandomForestClassifier 

def cross_val(data):
	k=5
	n= data.shape[0]
	X= data[:,:25]
	Y= data[:,25]
	err_train=[]
	err_test=[]
	# C_range = np.arange(start=9.8, stop=10.2, step=0.1)
	C_range=[1e0]
	# gamma_range = np.arange(start=0.01, stop=0.1, step=0.01)
	gamma_range=[1]
	for C in C_range:
		for gamma in gamma_range:
			model=SVC(C=C, gamma= gamma)
			i=0
			e_train=0
			e_test=0
			while i<k:
				a= int((n*i)/k)
				b= int((n*(i+1))/k)
				train_X= np.concatenate((X[0:a], X[b:n]))
				test_X= X[a:b]
				train_Y= np.concatenate((Y[0:a], Y[b:n]))
				test_Y= Y[a:b]
				model.fit(train_X,train_Y)
				pred_train=model.predict(train_X)
				pred_test=model.predict(test_X)
				e_train+=100*(1-metrics.accuracy_score(pred_train,train_Y))/k
				e_test+=100*(1-metrics.accuracy_score(pred_test,test_Y))/k
				i+=1
			err_train.append(e_train)
			err_test.append(e_test)
	# plt.plot(C_range, err_train, 'b')
	print(100.0-err_train[0])
	# plt.plot(C_range, err_test, 'r')
	# print(err_test)
	# plt.show()

def hyperparam_tuning(X, Y):
	kernel= ['linear', 'poly', 'rbf', 'sigmoid']
	degree= list(range(5))
	# C_range = np.logspace(-2 ,2, 5)
	C_range=[10]
	gamma_range=[1]
	# gamma_range = np.logspace(-2, 2, 5)
	# C_range = np.arange(start=10, stop=15, step=1)
	# gamma_range = np.arange(start=0.001, stop=0.01, step=0.001)
	param_grid = dict(C=C_range, gamma= gamma_range, kernel= [kernel[2]])
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
	grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
	grid.fit(X, Y)
	print("The best parameters are %s with a score of %0.5f"
		  % (grid.best_params_, grid.best_score_))

def scale(data):
	for i in range(25):
		data[:,i]= data[:,i]/np.amax(data[:,i])
	return data

if __name__ == "__main__":
	data= pd.read_csv('/home/ritik/Downloads/train_set.csv', names=list(range(1,27)))
	data= np.array(data)
	data= scale(data)
	# cross_val(data)
	hyperparam_tuning(data[:,:25], data[:,25])
	kernel= ['linear', 'poly', 'rbf', 'sigmoid']
	model= SVC(C=10, gamma=0.1)
	model.fit(data[:,:25], data[:,25])
	data= pd.read_csv('/home/ritik/Downloads/test_set.csv', names=list(range(1,26)))
	data= np.array(data)
	data= scale(data)
	pred= model.predict(data[:,:25])
	pred= pred.astype(int)
	# print(pred)
	z=range(2000)
	dict = {'id' : z, 'class' : pred}
	df = pd.DataFrame(dict)
	df.to_csv('pred.csv', index=False)