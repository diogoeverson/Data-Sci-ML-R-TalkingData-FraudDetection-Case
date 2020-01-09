#modelo de aprendizado de maquina para determinar clique fraudulento.
#Projeto Kaggle contendo data souce
#https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

#setup working directory
setwd("C:/FCD/BigDataRAzure/Cap20/Projeto01")
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

#PRE PROCESSAMENTO + #ANALISE EXPLORATORIA

#Importa dados para datasets - trabalhando somente com amostra dos dados de treino devido limitações do desktop

treino <- read.csv("train_sample.csv", header = TRUE, sep = ",")

head(treino)
View(treino)
summary(treino)
str(treino)
?dplyr

count(treino, ip)
count(treino, app)
count(treino, device)
count(treino, os)
count(treino, channel)
count(treino, click_time)
count(treino, attributed_time)
count(treino, as.factor(is_attributed))
count(treino, ip)


#Ajustes variáveis do tipo Fator e Data e hora
str(treino)
?factor

treino$click_time = as.Date(treino$click_time)
treino$attributed_time = as.Date(treino$attributed_time != NA)

str(treino)
View(treino)

#demais ajustes variáveis
?dplyr

distinct(treino, is_attributed)
#Gráficos para análise exploratória histograma e boxplot
?ggplot2

ggplot(treino, aes(app)) +
  geom_bar()

ggplot(treino, aes(device)) +
  geom_bar()

ggplot(treino, aes(os)) +
  geom_bar()

ggplot(treino, aes(channel)) +
  geom_bar()


hist(treino$is_attributed, main = "Histograma is_attributed")

boxplot(treino$is_attributed, main = "Boxplot is_attributed")

#obs. a variavel target esta desbalanceada na amostra de treino.


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

# Gerando previsÃµes nos dados de teste
previsoes <- data.frame(observado = as.factor(dados_teste$is_attributed),
                        previsto = predict(modelo, newdata = dados_teste))

# Visualizando o resultado
View(previsoes)
View(dados_teste)

#AVALIACAO

confusionMatrix(previsoes$observado, previsoes$previsto)

