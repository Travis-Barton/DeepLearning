#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:15:11 2018

@author: travisbarton
"""
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn import svm
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import spacy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from random import choice, sample
import warnings
from progress.bar import ChargingBar
warnings.simplefilter(action='ignore', category=FutureWarning)

nlp = spacy.load('en_vectors_web_lg')
RS = 69


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def Sub_treater(vec, sub):
    holder = []
    for i in range(len(vec)):
        if str(vec[i]) not in str(sub):
            #holder.append('Not_{}'.format(sub))
            holder.append('other')
        else:
            holder.append(str(vec[i]))
    return(holder)


def Binary_network(X, Y, X_test, label, val_split, nodes, epochs, batch_size, verbose = 0):
    model = Sequential()

    model.add(Dense(nodes, input_dim = X.shape[1], activation = 'linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(.4))
    model.add(Dense(nodes, activation = 'linear'))
    model.add(LeakyReLU(alpha = .001))
    model.add(Dense(2, activation = 'softmax'))        
            
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    #filepath="Best_{}.hdf5".format(label)
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
     #                            save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]

    model_history = model.fit(X, Y, 
                              epochs=epochs, batch_size=batch_size, 
                              verbose = verbose, validation_split = val_split)
    #physpreds = model.predict(X)
    #confm = confusion_matrix(Pred_to_num(Y), Pred_to_num(physpreds))
    #plot_confusion_matrix(confm, [0,1], normalize = True, title = "?")
    #print(X_test)
    if (X_test.ndim == 1):
        X_test = np.array([X_test])
    return([model.predict(X)[:,0], model.predict(X_test)[:,0]])
'''

def Binary_network(X, Y, X_test, label, val_split, nodes, epochs, batch_size, verbose = 0):
    model = Sequential()

    model.add(Dense(nodes, input_dim = X.shape[1], activation = 'linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(.5))
    model.add(Dense(nodes, activation = 'linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(.4))
    model.add(Dense(nodes, activation = 'linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dense(2, activation = 'softmax'))        
            
    model.compile(loss='binary_crossentropy', 
                  optimizer='sgd', 
                  metrics=['accuracy'])
    #filepath="Best_{}.hdf5".format(label)
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
     #                            save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]

    model_history = model.fit(X, Y, 
                              epochs=epochs, batch_size=batch_size, 
                              verbose = verbose, validation_split = val_split)
    physpreds = model.predict(X)
    confm = confusion_matrix(Pred_to_num(Y), Pred_to_num(physpreds))
    plot_confusion_matrix(confm, [0,1], normalize = True, title = "?")
    
    print(X_test)
    plt.figure()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
    plt.subplot(211)
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title("Accuracy")
    plt.xticks(range(20))
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
    plt.subplot(212)
    plt.title("\nLoss")
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.xticks(range(20))

    if (X_test.ndim == 1):
        X_test = np.array([X_test])
    return([model.predict(X)[:,0], model.predict(X_test)[:,0]])

'''

def Feed_reduction(X, Y, X_test, labels = None, val_split = .1, nodes = None, epochs = 15, batch_size = 30, verbose = 0):            
    if nodes == None:
        nodes = np.round(X.shape[0]/4)
    labels = np.unique(Y)
    onehot_encoder = OneHotEncoder(sparse=False) 
    finaltrain = np.empty([X.shape[0], len(labels)])
    finaltest = np.empty([X_test.shape[0], len(labels)])
    i = 0
    how_many = len(labels)
    bar = ChargingBar('Networks Loaded', max=how_many)
    for label in labels:
        
        x = X.copy()
        y = Y.copy()
        x_test = X_test.copy()
        y = Sub_treater(y, (label))
        y = pd.factorize(y)[0]
        y = y.reshape(len(y), 1).astype(int)        
        y = onehot_encoder.fit_transform(y)
        temp = Binary_network(x, y, x_test, label, val_split, nodes, epochs, batch_size, verbose)
        finaltrain[:,i] = temp[0]
        finaltest[:,i] = temp[1]
        bar.next()
        i +=1
    bar.finish()
    return([finaltrain, finaltest])
        


















