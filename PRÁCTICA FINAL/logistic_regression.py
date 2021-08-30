#!/usr/bin/env python
# coding: utf-8

# # IMPORTS

# In[1]:


import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from dataset_functions import *
from common_functions import *


# # MULTIVARIABLE LOGISTIC REGRESSION

# ## FUNCTIONS

# ### COST AND GRADIENT FUNCTIONS

# In[2]:


def coste(theta, X, Y, lam):
    """
    Función de coste, recibe como entrada las thetas, las características,
    los resultados y el parámetro lambda de regularización,
    devuelve el coste regularizado
    """
    m = X.shape[0]
    n = X.shape[1]

    h_theta = np.dot(X, theta)
    sig = sigmoid(h_theta)
    positive = np.dot(np.log(sig).T, Y)
    negative = np.dot(np.log(1 - sig).T, 1 - Y)
    J_theta = (-1 / m) * (positive + negative)
    
    # Regularizacion
    reg = (lam /(2 * m)) * np.sum(np.square(theta))
    
    # Coste Regularizado
    J_theta += reg
    
    return J_theta


# In[3]:


def gradiente(theta, X, Y, lam):
    """
    Función de gradiente, recibe como entrada las thetas, las características,
    los resultados y el parámetro lambda de regularización,
    devuelve la gradiente regularizada
    """
    m = X.shape[0]
    n = X.shape[1]
    
    h_theta = np.dot(X, theta.T)
    sig = sigmoid(h_theta)
    gradient = (1/m) * np.dot(sig.T - Y, X)
    
    # Regularizacion
    reg = (lam / m) * theta
    
    # Gradiente Regularizada
    gradient += reg
    
    return gradient


# ### PLOT LAMBDAS / COST FUNCTION

# In[4]:


def print_cost_lambdas(lambdas_list, costs_list):
    """
    Función que imprime las lambdas en el eje X,
    y el coste en el eje Y
    """
    plt.figure()
    plt.plot(lambdas_list, costs_list, c = 'r')
    plt.xlabel('lambda')
    plt.ylabel('cost')
    plt.show()


# ### PREDICTION FUNCTIONS

# In[5]:


def predict(theta, X, Y):
    """
    Función que, a partir de las thetas, las características,
    los resultados, calcula el porcentaje de acierto que tiene nuestra IA
    Imprime el porcentaje con dos decimales
    """
    predictions = np.dot(X, theta) > 0
    
    hits = np.sum(predictions == Y)
    percentage = hits / X.shape[0] * 100
    
    print("The logistic regression is reliable in {:.2f}% of the time\n".format(percentage))
    
    return percentage


# ### CHOOSE OPTIMAL VALUES FUNCTION

# In[2]:


def get_opt_thetas(theta, X, Y, reg = True):
    """
    Función que recibe como entrada las thetas, las características,
    los resultados y si se quiere aplicar un parámetro de regularización.
    Devuelve la theta optima con diferentes regularizaciones (en el caso de
    que reg = True)
    Además, llama a la función 'print_cost_lambdas' para comparar los
    distintos costes con las diferentes regularizaciones
    """
    # Inicializamos los valores
    lambdas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    n = X.shape[1]   
    
    theta_opt = np.zeros(n)
    lambd_opt = lambdas[0]
    cost_list = []
    
    
    # Probamos los distintos parámetros de regularización
    if reg:
        for lambd in lambdas:

            theta, _, _ = opt.fmin_tnc(
                func=coste,
                x0 = theta,
                fprime=gradiente,
                args=(X, Y, lambd)
            )
            
            actual_cost = coste(theta, X, Y, lambd)
            opt_cost = coste(theta_opt, X, Y, lambd_opt)


            # Coste actual vs Coste óptimo
            if (opt_cost > actual_cost):
                theta_opt = theta
                lambd_opt = lambd

            cost_list.append(actual_cost)
            
        # Dibujamos la evolución del coste respecto a la regularización
        print_cost_lambdas(lambdas, cost_list)
            
            
    else:
        theta_opt, _, _ = opt.fmin_tnc(
            func=coste,
            x0 = theta,
            fprime=gradiente,
            args=(X, Y, 0)
        )
        
    
    return theta_opt


# ## EXTERNAL FUNCTIONS

# In[7]:


def save_lr_model(thetas):
    """
    Guarda un array de thetas en la ruta models/theta_lr.npy
    """
    np.save("models/theta_lr.npy", thetas)


# In[8]:


def load_lr_model():  
    """
    Carga un array de thetas en la ruta models/theta_lr.npy
    """
    thetas = np.load("models/theta_lr.npy")
    return thetas


# In[9]:


def show_lr_prediction():
    """
    Carga los datos y las thetas óptimas, divide los datos y
    prueba las thetas óptimas sobre esos datos.
    Finalmente muestra el porcentaje de acierto de esos datos
    """
    X, Y = read_dataset()
    X, Y = manage_data(X, Y, use_onehot = False)
    _, _, _, _, X_test, Y_test = divide_dataset(X, Y)
    theta = load_lr_model()
        
    predict(theta, X_test, Y_test)


# In[10]:


def predict_example_lr(example):
    """
    Carga las theta óptimas y, a partir de un ejemplo, predice su resultado
    devuelve la predicción como booleano
    """
    theta = load_lr_model()
    prediction = np.dot(example, theta) > 0

    return prediction


# In[11]:


def main_lr():
    """
    Función que entrena el clasificador, obtiene las theta óptimas y
    guarda el modelo óptimo.
    """
    X, Y = read_dataset()
    X, Y = manage_data(X, Y, use_onehot = False)
    
    m = X.shape[0]
    n = X.shape[1]

    X_train, Y_train, _, _, X_test, Y_test = divide_dataset(X, Y)

    theta = np.zeros(n)
    
    # CLASIFICACIÓN SIN REGULARIZACIÓN
    print('LOGISTIC REGRESSION WITHOUT REGULARIZATION')
    theta_opt = get_opt_thetas(theta, X_train, Y_train, reg=False)
    predict_no_reg = predict(theta_opt, X_test, Y_test)
    
    # CLASIFICACIÓN CON REGULARIZACIÓN
    print('LOGISTIC REGRESSION WITH REGULARIZATION')
    theta_opt_reg = get_opt_thetas(theta, X_train, Y_train, reg=True)
    predict_reg = predict(theta_opt_reg, X_test, Y_test)
    
    if predict_no_reg > predict_reg:
        save_lr_model(theta_opt)
    else:
        save_lr_model(theta_opt_reg)


# In[12]:


# main_lr()


# In[ ]:




