#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 11:41:10 2019

@author: travisbarton
"""

## Reddit remade
import base64, datetime
import  praw, prawcore
import pandas as pd
import numpy as np
from collections import Counter
import datetime
import time
import requests
from Feed_network_maker import plot_confusion_matrix, Sub_treater, Binary_network, Feed_reduction
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


def Feed_reduction(X, Y, X_test, model_names = None, labels = None, val_split = .1, nodes = None, epochs = 15, batch_size = 30, verbose = 0, save = False):            
    if nodes == None:
            nodes = np.round(X.shape[1]/4).astype(int)
    if save == False:        
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
            temp = Binary_network(x, y, x_test, label, val_split, nodes, epochs, batch_size, verbose, model_names)
            finaltrain[:,i] = temp[0]
            finaltest[:,i] = temp[1]
            bar.next()
            i +=1
        bar.finish()
        return([finaltrain, finaltest])
    else:
        labels = np.unique(Y)
        onehot_encoder = OneHotEncoder(sparse=False) 
        finaltrain = np.empty([X.shape[0], len(labels)])
        i = 0
        how_many = len(labels)
        bar = ChargingBar('Networks Loaded', max=how_many)
        for label in labels:
            
            x = X.copy()
            y = Y.copy()
            x_test = X_test
            y = Sub_treater(y, (label))
            y = pd.factorize(y)[0]
            y = y.reshape(len(y), 1).astype(int)        
            y = onehot_encoder.fit_transform(y)
            temp = Binary_network(x, y, x_test, label, val_split, nodes, epochs, batch_size, None, verbose)
            finaltrain[:,i] = temp[0]
            bar.next()
            print(" network {} done.".format(i+1))
            i +=1
        bar.finish()
        return(finaltrain)
        
        
        



def Binary_network(X, Y, X_test, label, val_split, nodes, epochs, batch_size, model_name, verbose = 0):
    if model_name != None:
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
        model.load_weights("Best_{}.hdf5".format(label))
        if (X_test.ndim == 1):
            X_test = np.array([X_test])
        return([model.predict(X)[:,0], model.predict(X_test)[:,0]])
    else:
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
        filepath="Best_{}.hdf5".format(label)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, 
                                     save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
    
        model_history = model.fit(X, Y, 
                                  epochs=epochs, batch_size=batch_size, 
                                  verbose = verbose, validation_split = val_split,
                                  callbacks = callbacks_list)
        return([model.predict(X)[:,0]])        
    
def Sub_treater(vec, sub):
    holder = []
    for i in range(len(vec)):
        if str(vec[i]) not in str(sub):
            #holder.append('Not_{}'.format(sub))
            holder.append('other')
        else:
            holder.append(str(vec[i]))
    return(holder)

def Predict_post(dat, tags, Title):
    Title = nlp(Title).vector
    newdat = Feed_reduction(dat, tags, Title, model_names = "blah")
    clf = svm.SVC(kernel = 'linear')
    clf.fit(newdat[0], tags)
    #print(newdat[0].shape)
    #print(newdat[1].shape)
    pred = clf.predict(newdat[1])
    return(pred[0])


def main():
    reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='plLFnSdBy7b8ZQ', client_secret='_fv-EVVpz_m4iekd9a2EFsfJ66E',
                     username=base64.b64decode('UHJpdmF0ZUFza1NjaWVuY2VCb3Q='), 
                     password=(base64.b64decode("SUxvdmVMaW5kc2V5MTIz")))
    askscience = reddit.subreddit('askscience')
    subs = ['physics', 'bio', 'med', 'geo', 'chem', 'astro', 'eng']
    data = pd.read_csv(r'askscience_Data.csv')
    data = data.iloc[:,1:]
    
    history = pd.read_csv(r'history.csv')
    history = history.iloc[:, 1:]
    
    dat = np.empty([data.shape[0], 300])
    tags = Sub_treater(data.tag, subs)
    tags = [tag.replace('other', 'Other') for tag in tags]
    for i in range(data.shape[0]):
        temp = nlp(data.iloc[i,1]).vector
        for j in range(300):
            dat[i, j] = temp[j]
    print("Goodmorning General. I am loading the first round of networks, Sir!")
    Feed_reduction(dat, tags, X_test = None, model_names = None, save = True)
    
    print("\n General, my warmup is done, I am ready to begin my work!")
    i = 0
    while True:
        try: 
            for post in askscience.stream.submissions(skip_existing = True):
                data.loc[j,:] = [post.id, post.title, post.link_flair_css_class]
                data.to_csv("askscience_Data.csv")
                history = pd.read_csv(r'history.csv')
                history = history.iloc[:, 1:]   
                j = data.shape[0]
                i = history.shape[0]
                history.loc[i, 'actual'] = post.link_flair_css_class
                history.loc[i,'id'] = post.id
                history.loc[i, 'title'] = post.title
                history.loc[i, 'prediction'] = None
                pred = Predict_post(dat, tags, post.title)
                history.loc[i, 'prediction'] = pred
                if pred == post.link_flair_css_class:
                    history.loc[i, 'correct'] = 1
                    tags.append(post.link_flair_css_class)
                elif pred == 'Other' and post.link_flair_css_class not in tags:
                    history.loc[i, 'correct'] = 1
                    tags.append('Other')
                else:
                    history.loc[i, 'correct'] = 0
                    if post.link_flair_css_class in tags:
                        tags.append(post.link_flair_css_class)
                    else:
                        tags.append('Other')
                print("\n")
                history.loc[i, 'time'] = datetime.datetime.now().date()
                history.to_csv('history.csv')
                dat = np.vstack([dat, nlp(post.title).vector])
                if history.loc[i, 'correct'] == 1:
                    print("CORRECT!!!!!!!! New post #{}: {} \n with tag: {} and prediction {} \n My accuracy is now: {} \n".format(
                            history.shape[0],
                            post.title, 
                            post.link_flair_css_class, 
                            pred, 
                            round(sum(history['correct'])/history.shape[0], 4)*100))                                      
                else:
                    print("WRONG!!!!!!!!!! New post #{}: {} \n with tag: {} and prediction {} \n My accuracy is now: {} \n".format(
                            history.shape[0],
                            post.title, 
                            post.link_flair_css_class, 
                            pred, 
                            round(sum(history['correct'])/history.shape[0], 4)*100)) 
                i = i+1
                if i % 20 == 0:
                    print("Reloading networks, Sir. This may take a moment")
                    Feed_reduction(dat, tags, X_test = None, model_names = None, save = True)
        except Exception as e:
            print("I came accross an error general. I'll try restarting in 60 seconds: {} \n".format(e))
        time.sleep(60)
                    
            
        
    
    
    
main()    





    

    
    
    
    
    
    
    
    
    
    
    
