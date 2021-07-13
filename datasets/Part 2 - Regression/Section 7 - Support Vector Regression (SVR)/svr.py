#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:38:15 2019

@author: juangabriel
"""

### SVR

### Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

### Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

### Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

### Ajustar la regresión con el dataset
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
regression = SVR(kernel = 'poly', C = 100, gamma = 'auto', degree = 4, 
                 epsilon = .01, coef0 = 10)
regression.fit(X, y)
### Predicción de nuestros modelos con SVR
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[1]]))))
k = np.append(arr = sc_y.inverse_transform(regression.predict(X)).reshape(-1, 1), 
              values = sc_y.inverse_transform(y) , axis = 1)
mse = mean_squared_error(sc_y.inverse_transform(y), 
                         sc_y.inverse_transform(regression.predict(X)))
### Visualización de los resultados del SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_X.inverse_transform(X_grid), 
         sc_y.inverse_transform(regression.predict(X_grid)), 
         color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()