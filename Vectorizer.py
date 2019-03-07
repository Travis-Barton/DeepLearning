# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
This file translate the askscience_Data.csv into the numeric vectors. 
This is done with a proccess called word embeddings (https://spacy.io/usage/vectors-similarity)
Spacy and its models can be tricky to download, so heres a helpful link: https://spacy.io/usage/

'''

import spacy
import pandas as pd
import numpy as np
nlp = spacy.load('en_vectors_web_lg')
dat = pd.read_csv('askscience_Data.csv').iloc[:,2:]


def vectorizer(sent):
    return(nlp(sent).vector)

vectors = dat.iloc[:,1].apply(vectorizer)
vectors = vectors.values
vectors = np.vstack(vectors)
vectors2 = pd.concat([pd.DataFrame(vectors), dat.iloc[:,2]], axis = 1)

vectors2.to_csv("askscience_Data_Vectors.csv")
