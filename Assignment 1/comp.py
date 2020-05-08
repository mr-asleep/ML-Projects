import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from datetime import datetime
day={'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}

def read_data(data):
	dat=[]
	n= data.shape[0]
	for i in range(n):
	    a=[]
	    date= datetime.strptime(data.values[i][0], '%m/%d/%y')
	    a.append(float(date.month)) # x1
	    a.append(float(day[date.strftime("%A")]))# x2
	    a.append(float(date.year)) # x3
	    a.append(float((datetime.today().year - date.year)*12 + datetime.today().month - date.month)) # x4
	    a.append(data.values[i][1]) # x5
	    dat.append(a)
	d= np.array(dat)
	return d

def analytical_sol(a, m, l):
	n= a.shape[0]
	d= a[:, :m+1]
	Y=a[:,m+1]
	w= np.dot(np.dot(np.linalg.pinv(l*np.identity(m+1)+np.dot(d.T, d)), d.T), Y)
	y= np.dot(d, w)
	e= np.sum(np.square(y-Y))/n
	# print(e)
	df = pd.DataFrame({'Actual': Y, 'Predicted': y})
	# plt.scatter(a[:,2], a[:,23])
	# plt.scatter(a[:,2], np.dot(d,w))
	# plt.show()
	return w

# For getting regularisation parameter
def k_fold_validation_l(data, m): 
	k=4
	err_train=[]
	err_test=[]
	d= data[:,:m+1]
	Y_data= data[:,m+1]
	n= data.shape[0]
	l=8.8
	x=[]
	la=0
	e_min= 10.0
	while l<10:
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
		if e_test<e_min:
			e_min= e_test
			la=l
		x.append(l)
		l+=1e-1
	err_train= np.array(err_train)
	err_test= np.array(err_test)
	# plt.plot(x, err_train, 'b')
	# plt.plot(x, err_test, 'r')
	# plt.plot(x, np.absolute(err_test-err_train))
	# print(la)
	# plt.show()
	return la

# For predicting the output values on test data
def test_output(test, w):
	dat=[]
	for i in range(test.shape[0]):
	    a=[]
	    date= datetime.strptime(test.values[i][0], '%m/%d/%y')
	    a.append(float(date.month)) # x1
	    # a.append(float(datetime.today().year - date.year))
	    a.append(float(day[date.strftime("%A")]))# x2
	    a.append(float(date.year)) # x3
	    a.append(float((datetime.today().year - date.year)*12 + datetime.today().month - date.month)) # x4
	    dat.append(a)
	d= np.array(dat)
	data= np.array([[1, x1, x2, x1*x1, x2*x2, x1*x2, x1*x4, x2*x4, x1*x1*x2, x2*x2*x1, x1*x1*x1, x2*x2*x2 , x1*x1*x1*x1, x2*x2*x2*x2, math.sin(x1), math.sin(x2), math.sin(x1*x2), math.sin(x1*x1),math.sin(x2*x2), math.cos(x4), math.cos(x1*x4), math.cos(x2*x4), math.cos(x4*x4) ] for x1, x2, x3, x4 in d])
	t= np.dot(data, w)
	print(t)

if __name__ == "__main__":
	data= pd.read_csv('/home/ritik/Downloads/train.csv')
	d= read_data(data)
	data= np.array([[1, x1, x2, x1*x1, x2*x2, x1*x2, x1*x4, x2*x4, x1*x1*x2, x2*x2*x1, x1*x1*x1, x2*x2*x2 , x1*x1*x1*x1, x2*x2*x2*x2, math.sin(x1), math.sin(x2), math.sin(x1*x2), math.sin(x1*x1),math.sin(x2*x2), math.cos(x4), math.cos(x1*x4), math.cos(x2*x4), math.cos(x4*x4), x5] for x1, x2, x3, x4, x5 in d])
	l = k_fold_validation_l(data, 22)
	w= analytical_sol(data, 22, l)
	# print(w)
	test= pd.read_csv('/home/ritik/Downloads/test.csv')
	test_output(test, w)