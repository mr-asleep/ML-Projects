import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from assign import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPool2D, Input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.optimizers import RMSprop
from numpy import linalg as LA
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression

def pca_visualization(a, b):
	a= a.astype('float32')
	xu= a- np.mean(a, axis=0)
	S= np.dot(xu.T, xu)/a.shape[0]
	u, v = LA.eig(S)
	W= np.zeros((25, 784))
	for x, y in zip(a,b):
		x= x.reshape(-1,1)
		y= y.reshape(-1,1)
		W+= np.dot(y, np.linalg.pinv(x))
	W= W/3000.0
	a=[]
	for w in W:
		x= w/np.sqrt(np.sum(np.square(w)))
		a.append(x)
	a= np.array(a)
	for i, x in enumerate(a):
		ax = plt.subplot(5, 5, i+1)
		ax.set_xticks([])
		ax.set_yticks([])
		label= 'λ= '+str(u[i])
		ax.set_title(label)
		image= np.array(x).reshape(28,28).T
		plt.imshow(image, cmap='gray')
	plt.show()

def pca(X, Y):
	train_X, test_X, train_Y, test_Y= train_test_split(X, Y, test_size= 0.2,stratify=data[:,784], random_state=42)
	param= model(train_X,train_Y, 10, 0, [20])
	pred= predict(train_X, param)
	# # print(confusion_matrix(test_Y, pred))
	print('The accuracy of neural networks is',metrics.accuracy_score(pred,train_Y))

	# Accuracy with no hidden layer
	# model = LogisticRegression()
	# model.fit(train_X,train_Y)
	# prediction=model.predict(test_X)
	# print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y))

def CNN_visualization(model, x): # The expectation would be that the feature maps close to the input detect small or fine-grained detail, whereas feature maps close to the output of the model capture more general features.
	
	layer= 1
	# # Weights visualisations
	# filters, biases = model.layers[layer-1].get_weights()
	# # normalize filter values to 0-1 so we can visualize them
	# f_min, f_max = filters.min(), filters.max()
	# filters = (filters - f_min) / (f_max - f_min)
	# # plot first few filters
	# ix = 1
	# n_filters= 6
	# n_channels= 1
	# for i in range(n_filters):
	# 	# get the filter
	# 	f = filters[:, :, :, i]
	# 	# plot each channel separately
	# 	for j in range(n_channels):
	# 		# specify subplot and turn of axis
	# 		ax = plt.subplot(n_filters, n_channels, ix)
	# 		ax.set_xticks([])
	# 		ax.set_yticks([])
	# 		# plot filter channel in grayscale
	# 		plt.imshow(f[:, :, j], cmap='gray')
	# 		ix += 1
	# # show the figure
	# plt.show()

	# Feature maps visualization
	model = Model(inputs=model.inputs, outputs=model.layers[layer-1].output)
	sample= x[9]
	feature_maps = model.predict(sample.reshape(1, 28, 28, 1))
	filters = feature_maps.shape[-1]
	ax = plt.subplot(9, 9, 1)
	ax.set_xticks([])
	ax.set_yticks([])
	# plot filter channel in grayscale
	plt.imshow(sample.reshape(28, 28), cmap='gray')
	for i in range(filters):
		# specify subplot and turn of axis
		ax = plt.subplot(9, 9, i+2)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(np.squeeze(feature_maps[:,:, :, i]), cmap='gray')
	plt.show()

def CNN(X, Y):
	X = X.astype('float32')
	X= X/255.0
	x = X.reshape(X.shape[0], 28, 28, 1)
	y = to_categorical(Y, 10)
	train_X, test_X, train_Y, test_Y= train_test_split(x, y, test_size= 0.2,stratify=data[:,784], random_state=42)
	#model building
	d= np.arange(start=0.1, stop=0.9, step=0.1)
	for i in d:
		model = Sequential()
		model.add(Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape = (28,28,1)))
		model.add(MaxPool2D(pool_size=(2, 2)))
		model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'))
		model.add(MaxPool2D(pool_size=(2, 2)))    
		model.add(Dropout(i))
		model.add(Flatten())
		model.add(Dense(64, activation = "relu")) # Choose 128 for accuracy 96.5%
		model.add(Dense(10, activation = "softmax"))
		# Define the optimizer and compile the model
		model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
		# print (model.summary())
		history = model.fit(train_X, train_Y, epochs = 10, validation_data = (test_X, test_Y), verbose = 1)
	# CNN_visualization(model, test_X.reshape(-1,28,28))

def sparse_autoencoders(X, Y):
	train_X, test_X, train_Y, test_Y= train_test_split(X, Y, test_size= 0.2,stratify=data[:,784], random_state=42)
	train_X = train_X.astype('float32') / 255.0
	test_X = test_X.astype('float32') / 255.0
	n_h = 64
	input_img = Input(shape=(784,))
	code = Dense(n_h, activation='relu', activity_regularizer=regularizers.l1(10e-6))(input_img)
	output_img = Dense(784, activation='sigmoid')(code)
	modeL = Model(input_img, output_img)
	modeL.compile(optimizer='adam', loss='binary_crossentropy')
	history = modeL.fit(train_X, train_X, epochs=10)
	encoded = Model(input_img, code)
	# reconstructed = autoencoder.predict(test_X)
	weights = modeL.get_weights()[0].T
	# hidden_layer_vis(weights)

	train_X= encoded.predict(train_X)
	test_X= encoded.predict(test_X)
	param= model(train_X,train_Y, 10, 0, [20])
	pred= predict(test_X, param)
	# # print(confusion_matrix(test_Y, pred))
	print('The accuracy of neural networks is',metrics.accuracy_score(pred,test_Y))
	# Accuracy using the encoder
	# test_X= encoded.predict(test_X)
	# model = LogisticRegression()
	# model.fit(train_X,train_Y)
	# prediction=model.predict(test_X)
	# print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y))          

if __name__ == "__main__":

	data= pd.read_csv('/home/ritik/Downloads/2017EE10482.csv', header=None)
	data= np.array(data)
	X1= data[:,:784]
	Y= data[:, 784]
	CNN(X1, Y)
	# sparse_autoencoders(X1, Y)
	dat= pd.read_csv('/home/ritik/Downloads/pca.csv', header=None)
	dat= np.array(dat)
	X2= dat[:,:25]
	Y= dat[:, 25]
	# pca(X2,Y)
	# pca_visualization(X1, X2)
	