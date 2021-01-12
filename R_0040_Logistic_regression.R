model_rl <- glm(modelo, family = binomial(link='logit'),
                data = df_temp[index,])

confusionMatrix(factor(
  ifelse(
    predict(model_rl, type = 'response') > 0.5, "Anormal", "Normal")),
  df_temp[index,]$PATOLOGIA)

confusionMatrix(factor(
  ifelse(
    predict(model_rl, type = 'response', newdata = df_temp[-index,]) > 0.5, "Anormal", "Normal")),
  df_temp[-index,]$PATOLOGIA)

roc_rl <- roc(response = ifelse(df_temp[index,]$PATOLOGIA == "Normal", 0, 1),
              predictor = predict(model_rl, type = 'response'))

plot.roc(roc_rl,
         legacy.axes=T,
         print.auc=T,
         percent=T,
         col="#4daf4a")

ggplot(df_temp[index,]) + 
  geom_point(aes(x = Peso,
                 y = Altura,
                 col = df_temp[index,]$PATOLOGIA ==
                   ifelse(fitted(model_rl) > 0.7 ,"Anormal","Normal"))) + 
  theme(legend.position = 'bottom')

ggplot(df_temp[-index,]) + 
  geom_point(aes(x = Peso,
                 y = Altura,
                 col = df_temp[-index,]$PATOLOGIA ==
                   ifelse(predict(model_rl,
                                  newdata = df_temp[-index,]) > 0.5,"Anormal","Normal"))) + 
  theme(legend.position = 'bottom')