from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
#import tensorflow as tf
#from predict import predict

X_cats = np.ndarray((10,30000))
X_dogs = np.ndarray((10,30000))

def relu(x):
    x[x < 0] = 0
    return x
def drelu(x):
    return np.array(x>0,dtype = np.float32)

for i in range(10):
    img_cat = cv2.imread('Cats_1\img'+str(i)+'.jpg')
    img_cat = cv2.resize(img_cat, (100, 100), interpolation=cv2.INTER_AREA)
#    img_cat = cv2.cvtColor(img_cat, cv2.COLOR_BGR2GRAY)
    img_cat = np.asarray(img_cat)
    img_cat = img_cat.flatten()
    img_cat = img_cat.reshape((img_cat.shape[0],1))
    X_cats[i,:] = img_cat.T


    img_dog = cv2.imread('Dogs_1\img'+str(i)+'.jpg')
    img_dog = cv2.resize(img_dog, (100, 100), interpolation=cv2.INTER_AREA)
#    img_dog = cv2.cvtColor(img_dog, cv2.COLOR_BGR2GRAY)
    img_dog = np.asarray(img_dog)
    img_dog = img_dog.flatten()
    img_dog = img_dog.reshape((img_dog.shape[0], 1))
    X_dogs[i, :] = img_dog.T

X_cats = np.append(X_cats, np.zeros((X_cats.shape[0],1)), axis=1)
X_dogs = np.append(X_dogs, np.ones((X_cats.shape[0],1)), axis=1)

dataset = np.append(X_cats,X_dogs,axis=0)

dataset[:,0:dataset.shape[1]-1] = (dataset[:,0:dataset.shape[1]-1]-np.mean(dataset[:,0:dataset.shape[1]-1],axis=0))/np.std(dataset[:,0:dataset.shape[1]-1],axis=0)

np.random.shuffle(dataset)
data_train = dataset[0:int((dataset.shape[0]*0.7)),:]
data_test = dataset[data_train.shape[0]:dataset.shape[0],:]
data_cv = data_test[0:int((data_test.shape[0]*0.5)),:]
data_test = data_test[data_cv.shape[0]:data_test.shape[0],:]


#Separating the data into features and labels
X_train = data_train[:,0:-1]
X_cv = data_cv[:,0:-1]
X_test = data_test[:,0:-1]
y_train = np.ravel(data_train[:,[data_train.shape[1]-1]])
y_cv = np.ravel(data_cv[:,[data_cv.shape[1]-1]])
y_test = np.ravel(data_test[:,[data_test.shape[1]-1]])


classifier = MLPClassifier(hidden_layer_sizes=(5,5,5), max_iter=1000,activation = 'relu',solver='adam',random_state=1,alpha=0.00095)
classifier.fit(X_train, y_train)
y_pred_train = classifier.predict(X_train)
y_pred_cv = classifier.predict(X_cv)
y_pred_test = classifier.predict(X_test)
print('accuracy for training set is: '+ str(accuracy_score(y_train,y_pred_train)))
print('accuracy for cv set is: '+ str(accuracy_score(y_cv,y_pred_cv)))
print('accuracy for test set is: '+ str(accuracy_score(y_test,y_pred_test)))