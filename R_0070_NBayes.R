model_bayes <-
  e1071::naiveBayes(modelo,
                    data = df_temp[index,])

# confusion matrix of the training data
confusionMatrix(df_num[index,]$PATOLOGIA,
                predict(model_bayes, newdata = df_num[index,]))

# confusion matrix of the holdout data
confusionMatrix(df_num[-index,]$PATOLOGIA,
                predict(model_bayes, newdata = df_num[-index,]))

ggplot() + 
  # geom_point(aes(x = nd$x, y = nd$y)) + 
  geom_point(data = df_temp[index,],
             aes(x = Peso, 
                 y = Altura,
                 col = PATOLOGIA==predict(model_bayes,
                                          newdata = df_temp[index,]))) + 
  theme_light() + 
  labs(col = "correct?", title = "Naive Bayes")