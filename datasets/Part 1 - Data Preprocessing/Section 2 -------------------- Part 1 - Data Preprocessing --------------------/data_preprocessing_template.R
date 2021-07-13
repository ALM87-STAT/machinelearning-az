#### Plantilla para el Pre Procesado de Datos
setwd("~/GitHub/machinelearning-az/datasets/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------")

# Importar el dataset ####
dataset = read.csv('Data.csv')

# Tratamiento de los valores NA ####
dataset[is.na(dataset$Age), 2] <- mean(dataset$Age, na.rm = T)
dataset[is.na(dataset$Salary), 3] <- mean(dataset$Salary, na.rm = T)

# Codificar las variables categóricas ####
dataset$Country = factor(dataset$Country,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No", "Yes"),
                           labels = c(0,1))

# Dividir los datos en conjunto de entrenamiento y conjunto de test ####
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores ####
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[,2:3])

