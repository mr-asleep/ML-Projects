import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPool2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.optimizers import RMSprop
from numba import jit, cuda 

def data_visualization(X):
	result= X[5]
	# plt.imsave('filename.png', np.array(result).reshape(28,28).T, cmap=cm.gray)
	plt.imshow(np.array(result).reshape(28,28))
	plt.show()

def hidden_layer_vis(W1): #designed for 1 hidden layer
	a=[]
	for w in W1:
		x= w/np.sqrt(np.sum(np.square(w)))
		a.append(x)
	a= np.array(a)
	for i, x in enumerate(a):
		ax = plt.subplot(8, 8, i+1)
		ax.set_xticks([])
		ax.set_yticks([])
		image= np.array(x).reshape(28,28).T
		plt.imshow(image, cmap='gray')
	plt.show()

def one_hot_vector(Y):
	# n= Y.shape[0]
	# y= np.zeros((n,10))
	# for i in range(n):
	# 	y[i][Y[i]]=1
	y = np.array(to_categorical(Y, num_classes = 10))
	return y

def keras(X, Y): # in-built library
	Y= one_hot_vector(Y)
	train_X, test_X, train_Y, test_Y= train_test_split(X, Y, test_size= 0.2,stratify=data[:,784], random_state=42)
	model_1 = Sequential()
	# model_1.add(Dense(40, activation = "sigmoid"))
	model_1.add(Dense(40, activation = "sigmoid"))
	model_1.add(Dense(25, activation = "sigmoid"))
	# model_1.add(Dropout(0.5))
	model_1.add(Dense(10, activation = "softmax"))

	# Define the optimizer and compile the model
	# optimizer = optimizers.SGD(lr=0.5, clipnorm=5.)
	model_1.compile(optimizer= 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
	history = model_1.fit(train_X, train_Y, batch_size = 1, epochs = 10, validation_data = (test_X, test_Y), verbose = 1)


def activation(a, type):
	if type=='sigmoid':
		return 1/(1+np.exp(-a))
	if type=='tanh':
		return np.tanh(a)
	if type=='ReLu':
		return np.maximum(0,a)
	if type=='softmax':
		return np.exp(a)/np.sum(np.exp(a))

def derivative_act_fn(a, type):
	if type=='sigmoid':
		return np.multiply(a,1-a)
	if type=='tanh':
		return 1-np.power(a, 2)
	if type=='ReLu':
		deriv=[]
		for l in a:
			deriv.append([0 if x<=0 else 1 for x in l])
		deriv= np.array(deriv)
		return deriv

def parameter_initialize(n_x, n_h, n_y):
	n= len(n_h)
	if n==3:
		W1= np.random.randn(n_h[0], n_x)
		b1= np.zeros((n_h[0],1))
		W2= np.random.randn(n_h[1], n_h[0])
		b2= np.zeros((n_h[1],1))
		W3= np.random.randn(n_y, n_h[1])
		b3= np.zeros((n_y,1))
		return np.array([W1, W2, W3, b1, b2, b3])
	else:
		W1= np.random.randn(n_h[0], n_x)
		b1= np.zeros((n_h[0],1))
		W2= np.random.randn(n_y, n_h[0])
		b2= np.zeros((n_y,1))
		return np.array([W1, W2, b1, b2])

def feed_forward(X, param):
	n= param.shape[0]
	if n==6:
		[W1, W2, W3, b1, b2, b3]= param
		Z1=[]
		Z2=[]
		Z3=[]
		for x in X:
			x= np.reshape(x, (-1,1))
			a1= np.dot(W1, x)+b1
			z1= activation(a1, 'sigmoid')
			Z1.append(z1)
			a2= np.dot(W2, z1)+b2
			z2= activation(a2, 'sigmoid')
			Z2.append(z2)
			a3= np.dot(W3, z2)+b3
			z3= activation(a3, 'softmax')
			Z3.append(z3)
		Z1=np.array(Z1)
		Z2=np.array(Z2)
		Z3=np.array(Z3)
		Z1= np.squeeze(Z1, axis=2)
		Z2= np.squeeze(Z2, axis=2)
		Z3= np.squeeze(Z3, axis=2)
		return [Z1, Z2, Z3]

	else:
		[W1, W2, b1, b2]= param
		Z1=[]
		Z2=[]
		for x in X:
			x= np.reshape(x, (-1,1))
			a1= np.dot(W1, x)+b1
			z1= activation(a1, 'sigmoid')
			Z1.append(z1)
			a2= np.dot(W2, z1)+b2
			z2= activation(a2, 'softmax')
			Z2.append(z2)
		Z1= np.array(Z1)
		Z2= np.array(Z2)
		Z1= np.squeeze(Z1, axis=2)
		Z2= np.squeeze(Z2, axis=2)
		return [Z1, Z2]
			
def back_prop(X, Y, param, z):
	n= len(z)
	if n==3:
		[z1,z2,z3]=z
		[W1, W2, W3, b1, b2, b3]= param
		del3= z3-Y
		db3= np.sum(del3, axis=0, keepdims=True).T
		dW3= np.dot(del3.T, z2)
		del2= np.multiply(np.dot(del3, W3), derivative_act_fn(z2, 'sigmoid'))
		db2= np.sum(del2, axis=0, keepdims=True).T
		dW2= np.dot(del2.T, z1)
		del1= np.multiply(np.dot(del2, W2), derivative_act_fn(z1, 'sigmoid'))
		db1= np.sum(del1, axis=0, keepdims=True).T
		dW1= np.dot(del1.T, X)
		return np.array([dW1, dW2, dW3, db1, db2, db3])
	else:
		[W1, W2, b1, b2]= param
		[z1,z2]= z
		del2= z2-Y
		db2= np.sum(del2, axis=0, keepdims=True).T
		dW2= np.dot(del2.T, z1)
		del1= np.multiply(np.dot(del2, W2), derivative_act_fn(z1, 'sigmoid'))
		db1= np.sum(del1, axis=0, keepdims=True).T
		dW1= np.dot(del1.T, X)
		return np.array([dW1, dW2, db1, db2])

def model(X, Y, epoch, regularisation_param, n_h): # number of hidden layers can vary from 1 to 2
	n= X.shape[0]
	Y= one_hot_vector(Y)
	n_x= X.shape[1]
	n_y= Y.shape[1]   
	param= parameter_initialize(n_x, n_h, n_y)
	# learning_rate= 0.065 #for pca
	# learning_rate= 0.005 #for full batch, sigmoid
	learning_rate= 0.5 #for sgd, sigmoid
	# learning_rate= 0.1 #for sgd, ReLu
	# learning_rate= 0.1 #for sgd, tanh
	i=0
	for b in range(1,1+1):
		for it in range(int(n*epoch/b)):
			x= X[i:i+b]
			y= Y[i:i+b]
			z= feed_forward(x,param)
			# cost=-np.sum(np.multiply(y,np.log(z[1]))+ np.multiply(1-y,np.log(1-z[1])))
			# cost = np.squeeze(cost)
			# if(it%2400 == 0):
			# 	print('Cost after iteration# {:d}: {:f}'.format(it, cost))
			grad= back_prop(x, y, param, z)
			param= (1-learning_rate*regularisation_param)*param-learning_rate*grad
			i+=b
			if i>=n:
				i=0
	return param

def early_stopping(X, Y): #Came around 15-16 for sgd
						#designed for 1 hidden layer
	train_X, test_X, train_Y, test_Y= train_test_split(X, Y, test_size= 0.2,stratify=data[:,784], random_state=42)
	train_acc=[]
	test_acc=[]
	epochs=20
	Y= one_hot_vector(train_Y)
	for epoch in range(21):
		param= model(train_X, train_Y, epoch, 0, [25])
		train_acc.append(metrics.accuracy_score(predict(train_X, param),train_Y))
		test_acc.append(metrics.accuracy_score(predict(test_X, param),test_Y))
	plt.plot(range(21), train_acc, 'b', label= 'Training Accuracy')
	# print(test_acc)
	plt.plot(range(21), test_acc, 'r', label= 'Test Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()

def regularisation(X, Y): #Came around 4e-10 for sgd
						#designed for 1 hidden layer
	train_X, test_X, train_Y, test_Y= train_test_split(X, Y, test_size= 0.2,stratify=data[:,784], random_state=42)
	train_acc=[]
	test_acc=[]
	# l_range = np.logspace(-11 ,-8, 4)
	l_range = np.arange(start=1e-10, stop=6*1e-10, step=1e-10)
	for l in l_range:
		param= model(train_X, train_Y, 16, l, [25])
		train_acc.append(metrics.accuracy_score(predict(train_X, param),train_Y))
		print(metrics.accuracy_score(predict(test_X, param),test_Y), ' ', l)
		test_acc.append(metrics.accuracy_score(predict(test_X, param),test_Y))
	plt.plot(l_range, train_acc, 'b', label= 'Training Accuracy')
	# print(test_acc)
	plt.plot(l_range, test_acc, 'r', label= 'Test Accuracy')
	plt.xlabel('Regularisation parameter')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()

def predict(X, param):
	pred=[]
	z_h= feed_forward(X,param)
	for z in z_h[-1]:
		i=0
		m=z[0]
		for j in range(1,10):
			if m<z[j]:
				m=z[j]
				i=j
		pred.append(i)
	pred= np.array(pred)
	return pred

def missclassification(X, Y, pred):
	i=0
	for t1, t2 in zip(pred, Y):
		if t2==5:
			if t1!=t2:
				print(t1)
				image= np.array(X[i]).reshape(28,28).T
				plt.imshow(image, cmap='gray')
				plt.show()
		i+=1

if __name__ == "__main__":
	# data= pd.read_csv('/home/ritik/Downloads/2017EE10482.csv', header=None)
	data= pd.read_csv('/home/ritik/Downloads/train.csv', header=None)
	data= np.array(data)
	X= data[:,:784]/1000.0
	Y= data[:,784]
	# print(data.shape)
	# data_visualization(X)
	# keras(X,Y)
	train_X, test_X, train_Y, test_Y= train_test_split(X, Y, test_size= 0.2,stratify=data[:,784], random_state=42)
	# early_stopping(X, Y)
	# regularisation(X,Y)
	param= model(train_X,train_Y, 1, 0, [15])
	# hidden_layer= hidden_layer_vis(param[0])
	pred= predict(test_X, param)
	# missclassification(test_X, test_Y, pred)
	# print(confusion_matrix(test_Y, pred))
	print('The accuracy of neural networks is',metrics.accuracy_score(pred,test_Y))