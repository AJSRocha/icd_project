model_rl <- glm(modelo, family = binomial(link='logit'),
                data = df_temp[index,])
ct_rl_train <-
confusionMatrix(factor(
  ifelse(
    predict(model_rl, type = 'response') > 0.5, "Anormal", "Normal")),
  df_temp[index,]$PATOLOGIA)

ct_rl_test <-
confusionMatrix(factor(
  ifelse(
    predict(model_rl, type = 'response', newdata = df_temp[-index,]) > 0.5, "Anormal", "Normal")),
  df_temp[-index,]$PATOLOGIA)

roc_rl <- roc(response = ifelse(df_temp[index,]$PATOLOGIA == "Normal", 0, 1),
              predictor = predict(model_rl, type = 'response'))

# Plots nao usados

# plot.roc(roc_rl,
#          legacy.axes=T,
#          print.auc=T,
#          percent=T,
#          col="#4daf4a")



