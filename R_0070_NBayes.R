model_bayes <-
  e1071::naiveBayes(modelo,
                    data = df_temp[index,])

# confusion matrix of the training data
ct_nb_train <-
confusionMatrix(df_temp[index,]$PATOLOGIA,
                predict(model_bayes, newdata = df_temp[index,]))

# confusion matrix of the holdout data
ct_nb_test <-
confusionMatrix(df_temp[-index,]$PATOLOGIA,
                predict(model_bayes, newdata = df_temp[-index,]))

ggplot() + 
  # geom_point(aes(x = nd$x, y = nd$y)) + 
  geom_point(data = df_temp[index,],
             aes(x = Peso, 
                 y = Altura,
                 col = PATOLOGIA==predict(model_bayes,
                                          newdata = df_temp[index,]))) + 
  theme_light() + 
  labs(col = "correct?", title = "Naive Bayes")
