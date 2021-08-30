#!/usr/bin/env python
# coding: utf-8

# # IMPORTS

# In[1]:


import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from dataset_functions import *
from common_functions import *


# # NEURAL NETWORKS

# ## FUNCTIONS

# ### ARQUITECTURE OF NEURAL NETWORK

# In[2]:


def model(input_size, hidden_layer, num_labels):
    """
    A partir de tres números enteros (entrada, capas ocultas y etiquetas)
    genera una arquitectura de red neuronal con los valores de theta
    aleatorios.
    Devuelve un único array con todas las theta
    """
    eIni = 0.12

    Theta1_sh = (hidden_layer, input_size + 1)
    Theta2_sh = (num_labels, hidden_layer + 1)

    thetas_random = dict()

    thetas_random["Theta1"] = random_thetas(Theta1_sh, eIni)
    thetas_random["Theta2"] = random_thetas(Theta2_sh, eIni)

    th_random = np.concatenate(
        (
            np.ravel(thetas_random["Theta1"]),
            np.ravel(thetas_random["Theta2"])
        )
    )
    
    return th_random


# ### COST AND GRADIENT FUNCTIONS

# In[3]:


def coste(X, Y, w, lam):
    """
    Función de coste, recibe como entrada las características,
    los resultados, los pesos (las thetas) y el parámetro lambda
    de regularización.
    Devuelve el coste regularizado
    """
    m = X.shape[0]
    J_theta = 0    
    aux = 0
    _, _, _, _, h_theta = forward_prop(X, w)
    
    for i in range(m):
        aux +=np.sum(-Y[i] * np.log(h_theta[i])
                     - (1 - Y[i])* np.log(1 - h_theta[i]))
            
    
    J_theta = (1 / m) * aux
    
    # Regularizacion
    reg = 0

    for theta in w.keys():
        if "__" not in theta:
            reg += np.sum(np.square(w[theta][:, 1:]))

    reg *= (lam /(2 * m))

    # Coste Regularizado
    J_theta += reg
    
    return J_theta


# In[4]:


def gradiente(X, Y, w, lam):
    """
    Función de gradiente, recibe como entrada las características,
    los resultados, los pesos (las thetas) y el parámetro lambda
    de regularización.
    Devuelve la gradiente regularizada
    """
    m = X.shape[0]
    
    d = dict()
    d["delta1"] = np.zeros(w["Theta1"].shape)

    d["delta2"] = np.zeros(w["Theta2"].shape)
    a1, z2, a2, z3, h_theta = forward_prop(X, w)

    for i in range(m):
        # Calcular d2 y d3
        d3 = h_theta[i] - Y[i]                  # (10, )
        g_z2 = a2[i] * (1 - a2[i])              # (26, )
        d2 = np.dot(d3, w["Theta2"]) * g_z2     # (26, )
        #d2 = d2[1:]                             # (25, )
    
        # Actualizar deltas
        d["delta1"] += np.dot(d2[1:, np.newaxis], a1[i][np.newaxis, :])
        d["delta2"] += np.dot(d3[:, np.newaxis], a2[i][np.newaxis, :])

    d["delta1"] /= m
    d["delta2"] /= m
    
    #Regularizar deltas
    reg1 = ((lam / m) * w["Theta1"][:, 1:]) # No theta primera columna
    reg2 = ((lam / m) * w["Theta2"][:, 1:]) # No theta primera columna
    d["delta1"][:, 1:] += reg1              # j = 0 no tiene regularización
    d["delta2"][:, 1:] += reg2              # j = 0 no tiene regularización

    
    return np.concatenate(
        (np.ravel(d["delta1"]),
         np.ravel(d["delta2"]))
    )


# ### NEURAL NETWORK FUNCTIONS

# In[5]:


def forward_prop(X, w):
    """
    Función de propagación hacia adelante, aplica la lógica de las
    Redes Neuronales para únicamente dos theta.
    """
    a1 = X
    a1 = np.hstack([np.ones([X.shape[0], 1]), a1])

    z2 = np.dot(w['Theta1'], a1.T)
    a2 = sigmoid(z2).T
    a2 = np.hstack([np.ones([a2.shape[0], 1]), a2])

    z3 = np.dot(w['Theta2'], a2.T)
    h = sigmoid(z3).T

    return a1, z2, a2, z3, h


# In[6]:


def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, reg):
    """
    Esta función devuelve una tupla (coste, gradiente) con el coste y
    el gradiente de una red neuronal de tres capas, con num_entradas,
    num_ocultas nodos en la capa oculta y num_etiquetas nodos en la
    capa de salida. Si m es el número de ejemplos de entrenamiento,
    la dimensiónde de 'X' es (m, num_entradas) y la de 'y'
    es (m, num_etiquetas)
    """
    w = dict()
    w["Theta1"], w["Theta2"] = relocate(params_rn, num_entradas, num_ocultas, num_etiquetas)
    
    return coste(X, Y, w, reg), gradiente(X, Y, w, reg)


# ### SUPPORT FUNCTIONS

# In[7]:


def relocate(params_rn, num_entradas, num_ocultas, num_etiquetas):
    """
    A partir de un único vector y el número de entradas, de nodos ocultos
    y etiquetas se devuelven los vectores Theta1 y Theta2, con los tamaños
    establecidos
    """
    Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)],
                        (num_ocultas, (num_entradas + 1)))
    Theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):],
                        (num_etiquetas, (num_ocultas + 1)))
    return Theta1, Theta2


# In[8]:


def random_thetas(shape, E):
    """
    Recibe como parámetros las dimensiones y el Epsilon (rango -> [-E, E])
    
    Primero creamos una matriz de positivos y negativos
    Posteriormente creamos una matriz con números aleatorios positivos < Epsilon
    Multiplicamos las dos matrices, tenemos aleatorios positivos y negativos.
    
    Devuelve las dimensiones con valores aleatorios

    """
    posNeg = np.random.random((shape))
    pos = np.where(posNeg < .5)
    neg = np.where(posNeg >= .5)
    posNeg[pos] = 1
    posNeg[neg] = -1
        
    return (np.random.random((shape)) %E ) * posNeg


# ### TRAIN NEURAL NETWORK FUNCTION

# In[9]:


def train(backprop, thetas, X, Y, input_size, hidden_layer,
             num_labels, reg = 1, iterations = 70):
    """
    Función que, a partir de la función de backprop, las características,
    los resultados, los tamaños de las capas, la regularización y las
    iteraciones, entrena la red neuronal para obtener los parámetros theta
    óptimos.
    Devuelve las theta óptimas en forma de diccionario
    """
    
    # Thetas optimas
    res = opt.minimize(
        fun = backprop,
        x0 = thetas,
        args=(input_size, hidden_layer, num_labels, X, Y, reg),
        options={'maxiter': iterations},
        method='TNC',
        jac=True
    )

    # Recolocamos
    w = dict()
    w["Theta1"], w["Theta2"] = relocate(res.x,
                                         X.shape[1],
                                         hidden_layer,
                                         num_labels)

    return w


# ### PREDICTION FUNCTIONS

# In[10]:


def predict(X, w):
    """
    Función que a partir de las características y los pesos (thetas)
    predice los resultados de dichas características.
    Devuelve un array con esas predicciones
    """
    Y_hat = []
    _, _, _, _, pred = forward_prop(X, w)
    
    for i in range(pred.shape[0]):
        ejemplo = pred[i]
        num = np.argmax(ejemplo)
        Y_hat.append(num)
    
    Y_hat = np.array(Y_hat)
    
    return Y_hat


# In[11]:


def acc(X, Y, w):
    """
    Función que a partir de las características, los resultados
    y los pesos (thetas), calcula el porcentaje de acierto del clasificador.
    Devuelve un float con el porcentaje de acierto.
    """
    m = Y.shape[0]
    
    Y_hat = predict(X, w)
    
    percentage = np.round(
        np.sum(Y_hat == Y.argmax(1)) / m * 100,
        decimals = 2
    )
    
    return percentage


# ### PLOT ITERATIONS / COST FIGURE

# In[12]:


def print_opt_lambdas(iterations, hit_history, lambdas):
    """
    Función que imprime las iteraciones en el eje X,
    y la precisión en el eje Y
    """
    plt.figure()

    for hit in hit_history:
        plt.plot(iterations, hit)

    plt.xlabel('iterations')
    plt.ylabel('hit_accuracy')
    plt.legend(lambdas)
    
    plt.show()


# ### CHOOSE OPTIMAL VALUES FUNCTION

# In[13]:


def get_opt_thetas(th_random, X_train, Y_train, X_test, Y_test,
                    input_size, hidden_layer, num_labels):
    """
    Función que recibe unas theta aleatorias, las características y
    los resultados, tanto de train set como de test set, y el número de
    capas.
    Prueba, a partir de unas iteraciones y unos parámetros de
    regularización preestablecidos, todas las combinaciones, 
    de esta forma obtiene las theta óptimas (basándonos en la tasa de
    acierto que tiene)
    Devuelve la precisión, las thetas optimas y su combinación óptima
    de parámetros
    """
    iterations = [ 10, 50, 100, 200, 300 ]
    lambdas = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10 ]
#     iterations = [ 1,2,3 ]
#     lambdas = [ 0.01, 0.03, 0.1, 0.3 ]
    
    precision_opt = -1
    thetas_opt = np.array(0)
    it_opt = iterations[0]
    lambd_opt = lambdas[0]
    
    hit_lambda_history = []
    hit_total_history = []

    for lambd in lambdas:
        hit_lambda_history = []
        
        for i in iterations:
            thetas = train(backprop, th_random, X_train, Y_train,
                              input_size, hidden_layer, num_labels,
                              lambd, i)

            p = acc(X_test, Y_test, thetas)
            hit_lambda_history.append(p)
            
            print("Lambda: {}".format(lambd),
                  "\tIteraciones: {}".format(i),
                  "\tPrecisión:{}%\n".format(p)
                 )
            
            if(p > precision_opt):
                precision_opt = p
                thetas_opt = thetas
                it_opt = i
                lambd_opt = lambd
                
            if(precision_opt == 100.0):
                break
                

        hit_total_history.append(np.array(hit_lambda_history))
        
        if(precision_opt == 100.0):
            break

    if(precision_opt < 100.0):
        print_opt_lambdas(iterations, hit_total_history, lambdas)

    return precision_opt, thetas_opt, it_opt, lambd_opt


# ### STUDY BIAS AND VARIANCE

# In[14]:


def learning_errors(th_random,
                   num_entradas, num_ocultas, num_etiquetas,
                   X_train, Y_train, X_cv, Y_cv, lambd = 0,
                   iterations = 100, max_examples = 100
                  ):
    """
    A partir de unas theta aleatorias, las capas, las características
    y los resultados, tanto del train set como de la CV, la regularización,
    las iteraciones, y el número máximo de ejemplos a probar, calcula la
    tasa de error del trainset y de la CV set.
    Devuelve los vectores de error del entrenamiento y la CV
    """
    
    trainError = []
    cvError = []
    m = X.shape[0]
    
    for i in range(1, max_examples):
#         print(i, " de ", max_examples)
        X_i = X_train[0:i]
        Y_i = Y_train[0:i]

        res = opt.minimize(
            fun = backprop,
            x0 = th_random,
            args=(num_entradas, num_ocultas, num_etiquetas,
                  X_i, Y_i, lambd),
            method='TNC',
            jac=True,
            options={'maxiter': iterations},
        )

        trainError.append(backprop(res.x,
                                   num_entradas, num_ocultas, num_etiquetas,
                                   X_i, Y_i, lambd)
                          [0])
        cvError.append(backprop(res.x,
                                num_entradas, num_ocultas, num_etiquetas,
                                X_cv, Y_cv, lambd)
                       [0])
        
    return np.array(trainError), np.array(cvError)


# In[15]:


def print_learning_errors(train_error, cv_error):
    """
    Dibuja el error del entrenamiento y de la CV
    """
    
    plt.figure()
    
    r = range(0, len(train_error))

    plt.plot(r, train_error)
    plt.plot(r, cv_error)
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('Iterations')
    plt.ylabel('Cost')

    plt.show()


# ## EXTERNAL FUNCTIONS

# In[16]:


def save_nn_model(thetas):
    """
    Guarda dos array de thetas en la ruta models,
    con los archivos theta1_nn.npy  y  theta2_nn.npy
    """
    np.save("models/theta1_nn.npy", thetas["Theta1"])
    np.save("models/theta2_nn.npy", thetas["Theta2"])


# In[17]:


def load_nn_model():
    """
    Carga dos array de thetas en la ruta models,
    con los archivos theta1_nn.npy  y  theta2_nn.npy
    y los devuelve en un único diccionario
    """
    thetas = dict()
    
    thetas["Theta1"] = np.load("models/theta1_nn.npy")
    thetas["Theta2"] = np.load("models/theta2_nn.npy")
        
    return thetas


# In[18]:


def show_nn_prediction():
    """
    Carga los datos y las thetas óptimas, divide los datos y
    prueba las thetas óptimas sobre esos datos.
    Finalmente muestra el porcentaje de acierto de esos datos
    """
    X, Y = read_dataset()
    X, Y = manage_data(X, Y, use_onehot = True)
    _, _, _, _, X_test, Y_test = divide_dataset(X, Y)
    w = load_nn_model()
    
    print("The neural network is reliable in {:.2f}% of the time\n"
      .format(acc(X_test, Y_test, w)))
    


# In[19]:


def predict_example_nn(example):
    """
    Carga las theta óptimas y, a partir de un ejemplo, predice su resultado
    devuelve la predicción como booleano
    """
    w = load_nn_model()
    _, _, _, _, prediction = forward_prop(np.array([example]), w)
    
    return bool(prediction.argmin())


# In[20]:


def main_nn():
    """
    Función que entrena el clasificador, obtiene las theta óptimas y
    guarda el modelo óptimo.
    """
    X, Y = read_dataset()
    X, Y = manage_data(X, Y, use_onehot = True)

    m = X.shape[0]
    n = X.shape[1]

    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = divide_dataset(X, Y)
    
    input_size = X_train.shape[1]
    num_labels = 2

    hidden_layer_nodes = [ 10, 25, 50, 75, 100, 150, 200 ]

    acc_opt = 0
    thetas_opt = np.array(0)
    th_random_opt = np.array(0)
    it_opt = np.inf
    lambd_opt = np.inf
    hidden_layer_opt = hidden_layer_nodes[0]

    for nodes in hidden_layer_nodes: 

        th_random = model(input_size, nodes, num_labels)

        acc_act, thetas_act, it_act, lambd_act = get_opt_thetas(
            th_random,
            X_train,
            Y_train,
            X_test,
            Y_test,
            input_size,
            nodes,
            num_labels)

        if(acc_act > acc_opt):
            acc_opt = acc_act
            thetas_opt = thetas_act
            it_opt = it_act
            lambd_opt = lambd_act
            hidden_layer_opt = nodes

        if(acc_act == 100.0):
            break
        else:
            train_error, cv_error = learning_errors(np.concatenate((np.ravel(thetas_opt["Theta1"]), np.ravel(thetas_opt["Theta2"]))),
                    input_size, hidden_layer_opt, num_labels,
                    X_train, Y_train, X_cv, Y_cv,
                    lambd = lambd_opt, iterations = it_opt)

            print_learning_errors(train_error, cv_error)
        
    save_nn_model(thetas_opt)


# In[21]:


# main_nn()

