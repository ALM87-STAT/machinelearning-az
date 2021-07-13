# Apriori ####
setwd("~/GitHub/machinelearning-az/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori")

# Preprocesado de Datos ####
#install.packages("arules")
library(arules)
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep = ",", rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 50)

# Entrenar algoritmo Apriori con el dataset ####
rules = apriori(data = dataset, 
                parameter = list(support = 0.004, confidence = 0.2))
  
# Visualización de los resultados ####
inspect(sort(rules, by = 'lift')[1:10])
  
  
  