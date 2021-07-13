#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:07:43 2019

@author: juangabriel
"""

# Regresión Lineal Simple

### Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Importar el data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

### Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 1/3, 
                                                    random_state = 0)


### Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

### Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

### Predecir el conjunto de test
y_pred = regression.predict(X_test)

### Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

### Visualizar los resultados de test
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

### Otro código de trabajo para la Regresión Lineal Simple ###

import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

### Importar y explorar el data set 
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

dataset.shape
dataset.describe() 

### Gráfico exploratorio de los datos
dataset.plot(x='YearsExperience', y='Salary', style='o')  
plt.title('YearsExperience vs Salary')  
plt.xlabel('YearsExperience')  
plt.ylabel('Salary')  
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Salary'])

### Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 1/3, 
                                                    random_state = 0)

### Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#To retrieve the intercept:
print(regression.intercept_)
#For retrieving the slope:
print(regression.coef_)

### Predecir el conjunto de test
y_pred = regression.predict(X_test)

### Obervar la diferencia entre el valor real y el estimado
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

### Gráficar el valor real Vs el estimado
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

### Gráfico de la ecuación del modelo
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

