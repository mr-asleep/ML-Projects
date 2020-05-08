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

def binary_classfication(data, a, b):
	dat=[]
	for d in data:
		if d[25]==a or d[25]==b:
			dat.append(d)
	dat= np.array(dat)
	for i in range(dat.shape[0]):
		if dat[i][25]==a:
			dat[i][25]=-1
		else:
			dat[i][25]=1
	return dat

def cross_val(data):
	k=5
	n= data.shape[0]
	X= data[:,:25]
	Y= data[:,25]
	err_train=[]
	err_test=[]
	# C_range = np.arange(start=0.1, stop=1, step=0.1)
	C_range=[1]
	# gamma_range = np.arange(start=0.08, stop=0.15, step=0.01)
	gamma_range=[0.1]
	for C in C_range:
		for gamma in range(10):
			model=SVC(kernel='poly',C=C, gamma= 0.1, degree= gamma)
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
	plt.plot(range(10), err_train, 'b')
	# print(err_test)
	plt.plot(range(10), err_test, 'r')
	# print(err_test)
	plt.show()

def hyperparam_tuning(X, Y):
	kernel= ['linear', 'poly', 'rbf', 'sigmoid']
	# degree= list(range(5))
	C_range=[2]
	C_range = np.logspace(-2 ,3, 6)
	gamma_range = np.logspace(-3, 2, 6)
	# gamma_range=[0.1]
	# C_range = np.arange(start=0.04, stop=0.1, step=0.01)
	# gamma_range = np.arange(start=0.04, stop=0.1, step=0.01)
	param_grid = dict(C=C_range, gamma= gamma_range, degree= [3], kernel= [kernel[2]])
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
	grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
	grid.fit(X, Y)
	print("The best parameters are %s with a score of %0.5f"
		  % (grid.best_params_, grid.best_score_))

def kernel_matrix(X, gamma, kernel):
	K= np.zeros((X.shape[0], X.shape[0]))
	if kernel=='linear':
		for i, xi in enumerate(X):
			for j, xj in enumerate(X):
				K[i,j]= np.dot(xi, xj)
	else:
		for i, xi in enumerate(X):
			for j, xj in enumerate(X):
				K[i,j]= np.exp(-gamma * np.linalg.norm(xi - xj) ** 2)
	return K

def convex_opt(X, y,C,gamma, kernel):
	n= X.shape[0]
	y_matrix = y.reshape(1, -1)
	K= kernel_matrix(X, gamma, kernel)
	H = np.dot(y_matrix.T, y_matrix) * kernel_matrix(X, gamma, kernel)
	P = matrix(H)
	q = matrix(-np.ones((n, 1)))
	G = matrix(np.vstack((-np.eye((n)), np.eye(n))))
	h = matrix(np.vstack((np.zeros((n,1)), np.ones((n,1)) * C)))
	A = matrix(y_matrix)
	b = matrix(np.zeros(1))
	svm_parameters= solvers.qp(P, q, G, h, A, b)
	alphas= np.array(svm_parameters['x'])[:, 0]
	threshold = 1e-5 # Values greater than zero (some floating point tolerance)
	S = (alphas > threshold).reshape(-1, ) 
	# print(S.shape[0])  
	w = np.dot(K, alphas * y)
	b = y[S] - w[S]
	b = np.mean(b)
	pred= np.sign(w+b)
	print(b)
	print(metrics.accuracy_score(pred, y))
	return pred

def scale(data):
	for i in range(25):
		data[:,i]= data[:,i]/np.amax(data[:,i])
	return data

if __name__ == "__main__":
	data= pd.read_csv('/home/ritik/Downloads/2017EE10482.csv', names=list(range(1,27)))
	data= np.array(data)
	# data= scale(data)
	# print(np.amax(data[:,:25]))
	data= binary_classfication(data, 0, 1)
	train_X, test_X, train_Y, test_Y= train_test_split(data[:,:25], data[:,25], test_size= 0.2,stratify=data[:,25], random_state=42)
	kernel= ['linear', 'polynomial', 'rbf', 'sigmoid']
	# train_X= train_X[:,:10]
	# cross_val(data)
	# hyperparam_tuning(train_X, train_Y)
	# convex_opt(train_X, train_Y, 0.001, 0.01, 'linear')