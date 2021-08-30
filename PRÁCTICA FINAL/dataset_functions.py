#!/usr/bin/env python
# coding: utf-8

# # ESTE ARCHIVO CONTIENE FUNCIONES QUE PERMITEN LEER Y MODIFICAR EL FORMATO DE LAS 'X' E 'Y'

# In[1]:


import numpy as np

from pandas.io.parsers import read_csv
from sklearn.preprocessing import LabelEncoder


# In[2]:


def read_dataset():
    """
    Lee el dataset 'mushrooms.csv' y devuelve los datos
    divididos en X e Y
    """
    df = read_csv("mushrooms.csv")
    data = df.sample(frac = 1).to_numpy()          # SHUFFLE DATA

    X = np.array(data[:, 1:])
    Y = np.array(data[:, 0])
    
    return X, Y


# In[3]:


def onehot(Y):
    """
    A partir de un vector, normalmente de resultados, se aplica
    una codificación onehot y la devuelve en números enteros
    
    Por ejemplo:
    Y = [1, 2, 3, 1]
    
    Y_ONEHOT = [
        [1,0,0]
        [0,1,0]
        [0,0,1]
        [1,0,0]
    ]
    
    """
    
    le = LabelEncoder()
    labels = le.fit(Y[:]).classes_
    m = len(Y)
    
    Y = (Y - 1)
    Y_onehot = np.zeros((m, labels.shape[0]))
    
    for i in range(len(Y)):
        Y_onehot[i, int(Y[i])] = 1

    return Y_onehot.astype(int)


# In[4]:


def manage_data(X, Y, use_onehot = False):
    """
    Función que recibe como parámetros de entrada X, Y, y un booleano
    que transforma la Y a onehot para transformar los valores de dichos
    parámetros a int
    Devuelve la X y la Y en el formato deseado 
    """
    X = string2int(X, X.shape[1])    # OBLIGATORIO EXP CON FLOATS
    Y = string2int(Y)
    
    if use_onehot:
        Y = onehot(Y)

    X = np.hstack([np.ones([X.shape[0], 1], dtype=float), X])
    
    return X, Y


# In[5]:


def string2int(v, dim = 0):
    """
    Recibe como entrada un array de strings y las dimensiones que tiene,
    devuelve el mismo array con los strings cambiados a int, por ejemplo:
        ENTRADA:   ['A', 'B', 'C', 'A']
        SALIDA:    [0, 1, 2, 1]
    """
    le = LabelEncoder()

    if dim == 0:
        le.fit(v[:]).classes_      # 0 = EDIBLE
        v = le.transform(v)        # 1 = POISONOUS
        
    else:
        for i in range(0, dim):
            le.fit(v[:, i]).classes_
            v[:, i] = le.transform(v[:, i])
    
    return v.astype(float)


# In[6]:


def encode_example(example):
    """
    A partir de un ejemplo (Array de chars o strings) se devuelve
    un array de numeros codificados
    Por ejemplo:
        ENTRADA:   ['A', 'B', 'C', 'A']
        SALIDA:    [0, 1, 4, 0]
    """
    example = np.array([example])
    X, _ = read_dataset()
    le = LabelEncoder()
    

    for i in range(0, example.shape[1]):
        le.fit(X[:, i]).classes_
        example[:, i] = le.transform(example[:, i])
    
    example = np.hstack([1, example[0]])     # 'ONES COLUMN
    
    return example.astype(float).ravel()

# e = np.array(['x','s','n','t',
#               'p','f','c','n','k',
#               'e','e','s','s','w',
#               'w','p','w','o','p',
#               'k','s','u'])
# e = encode_example(e)
# print(e)


# In[7]:


def divide_dataset(X, Y, train_set = 0.8, cv_set = 0.1, test_set=0.1):
    """
    Divide el dataset con porcentajes que entran como parámetro, por ejemplo:
        Train_set = 80%
        Cv_set = 10%
        Test_set = 10%
    Devuelve el dataset dividido en Train set y Test set
    """

    train_set_size = int(train_set * X.shape[0])
    cv_set_size = int(cv_set * X.shape[0])

    X_train = X[:train_set_size, :]
    X_cv = X[train_set_size:train_set_size + cv_set_size, :]
    X_test = X[train_set_size + cv_set_size:, :]

    Y_train = Y[:train_set_size]
    Y_cv = Y[train_set_size:train_set_size + cv_set_size]
    Y_test = Y[train_set_size + cv_set_size:]
    
    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test

