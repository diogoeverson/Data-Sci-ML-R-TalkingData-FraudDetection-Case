#modelo de aprendizado de maquina para determinar clique fraudulento.
#Projeto Kaggle contendo data souce
#https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

#setup working directory
setwd("C:/FCD/BigDataRAzure/Cap20/Projeto01/Data-Sci-ML-R-TalkingData-FraudDetection-Case")
getwd()

#carga de pacotes
library(tidyr)
library(dplyr)
library(ggplot2)
library(caret) 
library(ROCR)
library(randomForest)
install.packages('e1071', dependencies=TRUE)
library(e1071)

#ANÁLISE EXPLORATORIA + PRÉ-PROCESSAMENTO
#(remoção de outliers, normalização(colocar os dados na mesma escala), redução de dimensionalidade(feature selection))

#Importa dados para datasets - trabalhando somente com amostra dos dados de treino devido limitações do desktop

treino <- read.csv("../train_sample.csv", header = TRUE, sep = ",")

head(treino)
View(treino)
summary(treino)

#ajustando categoria das variáveis
str(treino)

count(treino, ip)
treino$ip <- as.numeric(treino$ip)

count(treino, app) 
treino$app <- as.numeric(treino$app)

count(treino, device) 
treino$device <- as.numeric(treino$device)

count(treino, os) 
treino$os <- as.numeric(treino$os)

count(treino, channel) 
treino$channel <- as.numeric(treino$channel)

count(treino, click_time) 
treino$click_time <- as.numeric(treino$click_time)

count(treino, attributed_time) 
treino$attributed_time <- as.numeric(treino$attributed_time)

count(treino, as.factor(is_attributed)) #fator target
treino$is_attributed <- as.factor(treino$is_attributed)

str(treino)
head(treino)
#normalização das variáveis
?scale
treino$ip <- scale(treino$ip, center=T, scale=T)
treino$app <- scale(treino$app, center=T, scale=T)
treino$device <- scale(treino$device, center=T, scale=T)
treino$os <- scale(treino$os, center=T, scale=T)
treino$channel <- scale(treino$channel, center=T, scale=T)
treino$click_time <- scale(treino$click_time, center=T, scale=T)


# Dividindo os dados em treino e teste - 60:40 ratio
indexes <- sample(1:nrow(treino), size = 0.6 * nrow(treino))
indexes
train.data <- treino[indexes,]
test.data <- treino[-indexes,]

# Feature selection
?rfeControl
?rfe

control <- rfeControl(functions = rfFuncs, method = "cv", 
                      verbose = FALSE, returnResamp = "all", 
                      number = 3)

feature.vars = train.data[,-8]
class.var = train.data[,8]

variable.sizes <- 1:8

results.rfe <- rfe(x = feature.vars, y = class.var, 
                   sizes = variable.sizes, 
                   rfeControl = control)

# Visualizando os resultados
results.rfe
#?varImp
varImp((results.rfe))

# Criando e Avaliando o Modelo

# Construindo um modelo de regressão logística
formula.init <- "is_attributed ~ ."
formula.init <- as.formula(formula.init)

summary(treino)

lr.model <- glm(formula = formula.init, data = train.data, family = "binomial")

# Visualizando o modelo
summary(lr.model)

# Testando o modelo nos dados de teste
lr.predictions <- predict(lr.model, test.data, type="response")
lr.predictions <- round(lr.predictions)

# Avaliando o modelo
## separate feature and class variables
test.feature.vars <- test.data[,-8]
test.class.var <- test.data[,8]

confusionMatrix(table(data = lr.predictions, reference = test.class.var), positive = '1')














#APRENDIZADO
str(treino)

# Dividindo os dados em treino e teste - 70:30 ratio
indexes <- sample(1:nrow(treino), size = 0.7 * nrow(treino))
dados_treino <- treino[indexes,]
dados_teste <- treino[-indexes,]
# Verificando o numero de linhas
nrow(dados_treino)
nrow(dados_teste)

count(dados_treino, as.factor(is_attributed))
count(dados_teste, as.factor(is_attributed))

# Construindo o modelo
str(treino)

?randomForest

modelo <- randomForest(as.factor(is_attributed) ~ ip
                        + app
                        + device
                        + os
                        + channel
                        + click_time, 
                        data = dados_treino, 
                        ntree = 100, 
                        nodesize = 1,
                        na.action=na.exclude)


# Imprimondo o resultado
print(modelo)


#PREVISAO

# Gerando previsões nos dados de teste
previsoes <- data.frame(observado = as.factor(dados_teste$is_attributed),
                        previsto = predict(modelo, newdata = dados_teste))

# Visualizando o resultado
View(previsoes)
View(dados_teste)

#AVALIACAO

confusionMatrix(previsoes$observado, previsoes$previsto)

