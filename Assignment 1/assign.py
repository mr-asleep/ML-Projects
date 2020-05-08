import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Moore Penrose Inverse Solution
def analytical_solution(data, m, l):
	d= design_matrix(data, m)
	w= np.dot(np.dot(np.linalg.pinv(l*np.identity(m+1)+np.dot(d.T, d)), d.T), data.Y)
	y=np.dot(d,w)
	error= y-np.array(data.Y)
	e= np.sum(np.square(y-np.array(data.Y)))/2
	# plt.hist(error)
	plt.scatter(data.X, data.Y, label="Actual")
	plt.scatter(data.X, y, label="Predicted")
	plt.show()
	print (w)

def design_matrix(data, m):
	d=[]
	n= data.shape[0]
	for i in range(n):
		a=[]
		for j in range(m+1):
			a.append(pow(data.X.values[i], j))
		d.append(a)
	d= np.array(d)
	return d

# For getting order of the polynomial
def k_fold_validation_m(data): 
	k=4
	err_train=[]
	err_test=[]
	Y_data= np.array(data.Y)
	n= data.shape[0]
	for m in range(20):
		i=0
		e_train=0
		e_test=0
		d= design_matrix(data, m)
		while i<k:
			a= int((n*i)/k)
			b= int((n*(i+1))/k)
			d_train= np.concatenate((d[0:a], d[b:n]))
			d_test= d[a:b]
			train_y= np.concatenate((Y_data[0:a], Y_data[b:n]))
			test_y= Y_data[a:b]
			w= np.dot(np.dot(np.linalg.pinv(np.dot(d_train.T, d_train)), d_train.T), train_y)
			e_train+= np.sum(np.square(np.dot(d_train, w)-train_y))/(n*(k-1))
			e_test+= np.sum(np.square(np.dot(d_test, w)-test_y))/n
			i+=1
		err_train.append(e_train)
		err_test.append(e_test)
	plt.plot(err_train, 'b')
	plt.plot(err_test, 'r')
	# print(err_test)
	plt.show()
 
 # For getting regularisation parameter
def k_fold_validation_l(data, m): 
	k=4
	err_train=[]
	err_test=[]
	d= design_matrix(data, m)
	Y_data= np.array(data.Y)
	n= data.shape[0]
	l=0.005
	x=[]
	e=10.0
	la=0
	while l<0.3:
		i=0
		e_train=0
		e_test=0
		while i<k:
			a= int((n*i)/k)
			b= int((n*(i+1))/k)
			d_train= np.concatenate((d[0:a], d[b:n]))
			d_test= d[a:b]
			train_y= np.concatenate((Y_data[0:a], Y_data[b:n]))
			test_y= Y_data[a:b]
			w= np.dot(np.dot(np.linalg.pinv(l*np.identity(m+1)+np.dot(d_train.T, d_train)), d_train.T), train_y)
			e_train+= np.sum(np.square(np.dot(d_train, w)-train_y))/(n*(k-1))
			e_test+= np.sum(np.square(np.dot(d_test, w)-test_y))/n
			i+=1
		err_train.append(e_train)
		err_test.append(e_test)
		if e_test<e:
			e=e_test
			la=l
		x.append(l)
		l+=1e-2
	plt.plot(x, err_train, 'b')
	plt.plot(x, err_test, 'r')
	print(la)
	plt.show()

# Gradient Descent
def gradient_descent(data, m, q):
	n= data.shape[0]
	d= design_matrix(data, m)
	error=[]
	for b in range(n,n+1):
		a=6.25*1e-6
		w= np.zeros(m+1)
		i=0
		for it in range(10000):
			if i+b==n+1:
				i=0
			batch= d[i:i+b]
			# grad= np.dot(batch.T, (np.dot(batch,w)-data.Y[i:i+b])) 
			grad= np.dot(batch.T, (q)*(np.power(np.absolute(np.dot(batch,w)-data.Y), q-1)*np.sign(np.dot(batch,w)-data.Y)))
			w= w - a*grad
			y= np.dot(d, w)
			e= np.sum(np.power(np.absolute(y-np.array(data.Y)), q))/n
			# print(e)
			i+=1
			error.append(e)
	print(e)
	# plt.plot(error)
	# plt.scatter(data.X, data.Y, label="Actual")
	# plt.scatter(data.X, y, label="Predicted")
	# plt.show()
	# plt.scatter(data.X, np.dot(d, w))
	# error= y-np.array(data.Y)
	# plt.hist(error)
	# plt.show()
	return e

if __name__ == "__main__":
	data= pd.read_csv('/home/ritik/Downloads/Gaussian_noise.csv', names=['X', 'Y'])
	# data= data.head(20)
	# k_fold_validation_l(data, 8) # Can be used to get m
	# w= gradient_descent(data, 8, 2)
	# analytical_solution(data, 8, 0)
	k_fold_validation_m(data)

