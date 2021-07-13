# Plantilla para el Pre Procesado de Datos - Datos faltantes
# Importar el dataset
dataset = read.csv('Data.csv')


# Tratamiento de los valores NA
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

# Otra forma de tratamiento de los NAs
dataset[is.na(dataset$Age), 2] <- mean(dataset$Age, na.rm = T)
dataset[is.na(dataset$Salary), 3] <- mean(dataset$Salary, na.rm = T)

