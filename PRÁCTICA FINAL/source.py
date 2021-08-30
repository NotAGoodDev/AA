#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time

from dataset_functions import *
from common_functions import *
from logistic_regression import *
from neural_network import *
from svm import *


# In[ ]:


def manage_example_options(text, correct_answers):
    """
    A partir de un texto y un array de respuestas, valida la entrada
    recibida.
    Devuelve la opción seleccionada sabiendo que está entre las respuestas
    esperadas.
    """
    opt = None
    while(True):
        print(text)
        
        opt = input().lower()

        if opt in correct_answers:
            break 
        else:
            print('----- WRONG TYPE -----')
    
    return opt


# In[ ]:


def example_menu():
    """
    Función que muestra un menú por pantalla, y espera a una entrada.
    Cuando la respuesta esperada es correcta se añade a un array, que
    acabará siendo el ejemplar de hongo/seta.
    Devuelve dicho array
    """
    example = []
    
    text = """
    Cap shape?
        b: bell
        c: conical
        x: convex
        f: flat
        k: knobbed
        s: sunken
    """
    options = ['b', 'c', 'x', 'f', 'k', 's']
    example.append(manage_example_options(text, options))
    
    
    text = """
    Cap surface?
        f: fibrous
        g: grooves
        y: scaly
        s: smooth
    """    
    options = ['f', 'g', 'y', 's']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Cap color?
        n: brown
        b: buff
        c: cinnamon
        g: gray
        r: green
        p: pink
        u: purple
        e: red
        w: white
        y: yellow
    """
    options = ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Bruises?
        t: true
        f: false
    """
    options = ['t', 'f']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Odor?
        a: almond
        l: anise
        c: creosote
        y: fishy
        f: foul
        m: musty
        n: none
        p: pungent
        s: spicy
    """
    options = ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Gill attachment?
        a: attached
        f: free
    """
    options = ['a', 'f']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Gill spacing?
        c: close
        w: crowded
    """
    options = ['c', 'w']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Gill size?
        b: broad
        n: narrow
    """
    options = ['b', 'n']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Gill color?
        k: black
        n: brown
        b: buff
        h: chocolate
        g: gray
        r: green
        o: orange
        p: pink
        u: purple
        e: red
        w: white
        y: yellow
    """
    options = ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Stalk shape?
        e: enlarging
        t: tampering
    """
    options = ['e', 't']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Stalk root?
        b: bulbous
        c: club
        e: equal
        r: rooted
        ?: missing
    """
    options = ['b', 'c', 'e', 'r', '?']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Stalk surface above ring?
        f: fibrous
        y: scaly
        k: silky
        s: smooth

    """
    options = ['f', 'y', 'k', 's']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Stalk surface below ring?
        f: fibrous
        y: scaly
        k: silky
        s: smooth

    """
    options = ['f', 'y', 'k', 's']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Stalk color above ring?
        n: brown
        b: buff
        c: cinnamon
        g: gray
        o: orange
        p: pink
        e: red
        w: white
        y: yellow
    """
    options = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Stalk color below ring?
        n: brown
        b: buff
        c: cinnamon
        g: gray
        o: orange
        p: pink
        e: red
        w: white
        y: yellow
    """
    options = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Veil type?
        p: partial
    """
    options = ['p']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Veil color?
        n: brown
        o: orange
        w: white
        y: yellow
    """
    options = ['n', 'o', 'w', 'y']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Ring number?
        n: none
        o: one
        t: two
    """
    options = ['n', 'o', 't']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Ring type?
        e: evanescent
        f: flaring
        l: large
        n: none
        p: pendant
    """
    options = ['e', 'f', 'l', 'n', 'p']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Spore print color?
        k: black
        n: brown
        b: buff
        h: chocolate
        r: green
        o: orange
        u: purple
        w: white
        y: yellow
    """
    options = ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Population?
        a: abundant
        c: clustered
        n: numerous
        s: scattered
        v: several
        y: solitary
    """
    options = ['a', 'c', 'n', 's', 'v', 'y']  
    example.append(manage_example_options(text, options))
    
    
    text = """
    Habitat?
        g: grasses
        l: leaves
        m: meadows
        p: paths
        u: urban
        w: waste
        d: woods
        
    """
    options = ['g', 'l', 'm', 'p', 'u', 'w', 'd']  
    example.append(manage_example_options(text, options))
    
    return example


# In[ ]:


def print_mushroom_state(state):
    """
    A partir de un booleano, muestra por pantalla si el ejemplar de hongo
    puede ser ingerido o no
    """
    if(state):
        print("WARNING! THE MUSHROOM IS POISONOUS")
    else:
        print("You can eat the mushroom safely\n\n")


# In[ ]:


def manage_lr():
    """
    Función que gestiona el clasificador de regresión logística.
    Pregunta si se puede reentrenar o si se quiere introducir manualmente
    un ejemplo.
    En el caso de querer reentrenar el clasificador, se cronometrará y
    se llamará a la función main_lr().
    En el caso de querer probar un ejemplar, se llamará a la función
    example_menu() y posteriormente predecirá si puede ser comestible o no.
    """
    opt = input("Do you want to retrain the LR? (y/n)")
    
    if(opt.lower() == 'y'):
        print('----- RETRAINING LR -----')
        tic = time.process_time()
        main_lr()
        toc = time.process_time()
        print("TIME: {} seconds".format(toc - tic))
    
    opt = input("Do you want to try an example? (y/n)")

    if(opt.lower() == 'y'):
        e = example_menu()
        e = encode_example(e)
        state = predict_example_lr(e)
        print("REMEMBER!")
        show_lr_prediction()
        print_mushroom_state(state)


# In[ ]:


def manage_nn():
    """
    Función que gestiona el clasificador de la red neuronal.
    Pregunta si se puede reentrenar o si se quiere introducir manualmente
    un ejemplo.
    En el caso de querer reentrenar el clasificador, se cronometrará y
    se llamará a la función main_nn().
    En el caso de querer probar un ejemplar, se llamará a la función
    example_menu() y posteriormente predecirá si puede ser comestible o no.
    """
    opt = input("Do you want to retrain the NN? (y/n)")
    if(opt.lower() == 'y'):
        print('----- RETRAINING NN -----')
        tic = time.process_time()
        main_nn()
        toc = time.process_time()
        print("TIME: {} seconds".format(toc - tic))
        
    opt = input("Do you want to try an example? (y/n)")

    if(opt.lower() == 'y'):
        e = example_menu()
        e = encode_example(e)
        state = predict_example_nn(e)
        print("REMEMBER!")
        show_nn_prediction()
        print_mushroom_state(state)


# In[ ]:


def manage_svm():
    """
    Función que gestiona el clasificador de la SVM.
    Pregunta si se puede reentrenar o si se quiere introducir manualmente
    un ejemplo.
    En el caso de querer reentrenar el clasificador, se cronometrará y
    se llamará a la función main_svm().
    En el caso de querer probar un ejemplar, se llamará a la función
    example_menu() y posteriormente predecirá si puede ser comestible o no.
    """
    opt = input("Do you want to retrain the SVM? (y/n)")
    if(opt.lower() == 'y'):
        print('----- RETRAINING SVM -----')
        tic = time.process_time()
        main_svm()
        toc = time.process_time()
        print("TIME: {} seconds".format(toc - tic))

    opt = input("Do you want to try an example? (y/n)")

    if(opt.lower() == 'y'):
        e = example_menu()
        e = encode_example(e)
        state = predict_example_svm(e)
        print("REMEMBER!")
        show_svm_prediction()
        print_mushroom_state(state)


# In[ ]:


def menu():
    """
    Menu principal del programa, permite escoger un tipo de clasificador
    o salir del programa
    """
    
    print("Welcome to the final subject project - MUSHROOM CLASSIFIER")
    
    while (True):
        print(
    """
    You can choose:

        1.\t\t\tLogistic Regression Classifier
        2.\t\t\tNeural Network Classifier
        3.\t\t\tSupport Vector Machine Classifier
        0 or other key.\t\tExit
    """
        )
              
        
        opt = input("Write the number or the first letter: ")

        switcher = {
            'L': 1,
            'N': 2,
            'S': 3,
            'E': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '0': 0
        }

        opt = switcher.get(opt.upper(), 0)
        
        if opt == 0:
            break
        elif opt == 1:
            manage_lr()
        elif opt == 2:
            manage_nn()
        else:
            manage_svm()


# In[ ]:


menu()

