#!/usr/bin/env python
# coding: utf-8

# # ESTE ARCHIVO CONTIENE FUNCIONES COMUNES DE LOS CLASIFICADORES

# In[2]:


import numpy as np


# In[1]:


def sigmoid(z):
    """
    Función sigmoide
    recibe como entrada un número / array
    devuelve la función sigmoide
    """
    return 1 / (1 + np.exp(-z))

