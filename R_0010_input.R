
library(dplyr)
library(ggplot2)
library(e1071)

df_core <- readxl::read_xls("UCMF.xls")

# corrigimos nome das variaveis
names(df_core) <- c("ID","Peso","Altura",
                    "IMC","Atendimento","DN",
                    "IDADE","Convenio","PULSOS",
                    "PA_SISTOLICA","PA_DIASTOLICA","PPA",
                    "PATOLOGIA","B2","SOPRO",
                    "FC","HDA1","HDA2",
                    "SEXO","MOTIVO1","MOTIVO2")

# rejeitamos observaçoes sem variavel resposta
df <- df_core %>% filter(!is.na(PATOLOGIA))

## corrigimos os niveis da variavel
df$PATOLOGIA <- factor(df$PATOLOGIA)
levels(df$PATOLOGIA) <- c("Anormal","Anormal","Normal","Normal")

# IDADES
df <- df[df$IDADE > 0 & df$IDADE <= 20 & !is.na(df$IDADE),]
df$IDADE_class <- trunc(df$IDADE)

# SEXO
df$SEXO <- factor(df$SEXO)
levels(df$SEXO) <- c("F","F","I","M","M","M")

# Decisão: rejeitar sexo I
df <- df[df$SEXO != "I",]



df$HDA1[is.na(df$HDA1)] <- "Sem Historico"
df$HDA1 <- factor(df$HDA1) 
levels(df$HDA1)

df$HDA2[is.na(df$HDA2)] <- "Sem Historico"
df$HDA2 <- factor(df$HDA2)
levels(df$HDA2)

df$SOPRO <- factor(df$SOPRO)
levels(df$SOPRO) <- c("ausente", "presente", "presente",
                      "presente","presente","presente",
                      "presente")





# Peso e Altura













table(df$Peso)
table(df$Altura)

df[df$Peso > 100,] %>% View

ggplot(df) + 
  geom_point(aes(x = IDADE, y = Peso))
ggplot(df) + 
  geom_point(aes(x = IDADE, y = Peso))

ggplot(df[df$Altura!=0 & df$Peso!=0,]) + 
  geom_point(aes(x = Peso, y = Altura, col = IDADE))


df_input <- caret::preProcess(df[,c("IDADE","Altura","Peso","PA_SISTOLICA","PA_DIASTOLICA")], method='knnImpute')

predict(df_input, newdata = df[,c("IDADE","Altura","Peso","PA_SISTOLICA","PA_DIASTOLICA")])

par(mfrow = c(1,2))
hist(df$PA_DIASTOLICA)
hist(df[df$PA_SISTOLICA<700,]$PA_SISTOLICA)

ggplot(df[df$PA_SISTOLICA < 700,]) + 
  geom_histogram(aes(x = PA_SISTOLICA)) + 
  facet_wrap(IDADE~.)


