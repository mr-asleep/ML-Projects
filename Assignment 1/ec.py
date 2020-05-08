import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def analytical_solution(data, m, l):
	d= design_matrix(data, m)
	w= np.dot(np.dot(np.linalg.pinv(l*np.identity(m+1)+np.dot(d.T, d)), d.T), data.Y)
	y=np.dot(d,w)
	error= (y-np.array(data.Y))
	e= np.sum(np.square(y-np.array(data.Y)))/100.0
	# print(w)
	# print(min(error))
	# plt.hist(error)
	plt.scatter(data.X, data.Y, label="Actual")
	plt.scatter(data.X, y, label="Predicted")
	# # plt.scatter(data.X, y-np.array(data.Y))
	plt.show()
	# print(pd.DataFrame(np.sort(error)[40:]))
	# plt.scatter(data.X, data.Y, label="Actual")
	# plt.scatter(data.X, y, label="Predicted")
	# plt.show()
	# print(e)
	# error= np.absolute(y-np.array(data.Y))
	# df= pd.DataFrame({'Actual':data.Y, 'Predicted': y, 'Error': error})
	# print(df)
	# print(np.var(error), e)
	# print(np.sum(np.square(error))/100.0)
	return e

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

def k_fold_validation_m(data): # For getting order of the polynomial
	k=4
	err_train=[]
	err_test=[]
	Y_data= np.array(data.Y)
	n= data.shape[0]
	for m in range(13):
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
	plt.plot(range(7,13), err_train[7:], 'b')
	plt.plot(range(7,13), err_test[7:], 'r')
	plt.show()
 
def k_fold_validation_l(data, m): # For getting regularisation parameter
	k=4
	err_train=[]
	err_test=[]
	d= design_matrix(data, m)
	Y_data= np.array(data.Y)
	n= data.shape[0]
	l=0.005
	x=[]
	e=100.0
	la=1e-4
	while l<0.06:
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
			# print(la)
		x.append(l)
		l+=1e-5
	plt.plot(x, err_train, 'b')
	plt.plot(x, err_test, 'r')
	# plt.plot(np.absolute(np.subtract(err_train, err_test)))
	print(la, min(err_test))
	plt.show()

if __name__ == "__main__":
	data= pd.read_csv('/home/ritik/Downloads/NonGaussian_noise.csv', names=['X', 'Y'])
	# print(data)
	# plt.scatter(data.X, data.Y)
	# plt.show()
	# error=[]
	# for m in range(10):
	# 	error.append(gradient_descent(data, m, 0))
	# plt.plot(error)
	# plt.show()
	# k_fold_validation_m(data) # Can be used to get m
	analytical_solution(data, 9, 0)
	# k_fold_validation_l(data, 9)
