model_lda <- MASS::lda(modelo, data = df_temp[index,])

MASS::ldahist(predict(model_lda, data = df_temp[index,])$x[,1],
              g = predict(model_lda, data = df_temp[index,])$class)

lda_probs <- predict(model_lda, 
                     newdata = df_temp[index,], 
                     type = "response")

# model performance, quick glance
mean(ifelse(predict(model_lda, 
                    newdata = df_temp[index,])$x[,1] > 
              model_lda$prior[1] , 1, 0) == 
       ifelse(df_temp[index,]$PATOLOGIA == "Normal",0,1))

mean(ifelse(predict(model_lda,
                newdata = df_temp[-index,])$x[,1] >
                model_lda$prior[1] , 1, 0) ==
ifelse(df_temp[-index,]$PATOLOGIA == "Normal",0,1))

# confusion matrix
ct_lda_train <-
confusionMatrix(
  predict(model_lda)$class,
  df_temp[index,]$PATOLOGIA)

ct_lda_test <-
confusionMatrix(
  predict(model_lda, newdata = df_temp[-index,])$class,
  df_temp[-index,]$PATOLOGIA)

# Plot da decision boundary
# ggplot() + 
#   # geom_point(aes(x = nd$x, y = nd$y)) + 
#   geom_point(data = df_temp[index,],
#              aes(x = PA_DIASTOLICA, 
#                  y = PA_SISTOLICA,
#                  col = PATOLOGIA==predict(model_lda)$class)) + 
#   theme_light() + 
#   labs(col = "correct?")
