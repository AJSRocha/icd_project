grelha <- expand.grid(mtry = seq(3,12,3))

forest <- caret::train(modelo,
                       method="rf",
                       data=df_temp[index,],
                       trControl=control,
                       tuneGrid=grelha,
                       preProc = c("center","scale"),
                       ntree=500,
                       metric="Accuracy")
ct_rf_train <-
# confusion matrix of the training data
caret::confusionMatrix(forest)

# confusion matrix of the holdout data
ct_rf_test <-
confusionMatrix(df_temp[-index,]$PATOLOGIA,
                predict(forest, newdata = df_temp[-index,]))

var_rank<-data.frame(variables=rownames(forest$finalModel$importance),importance=forest$finalModel$importance)

var_rank[order(var_rank$MeanDecreaseGini,decreasing=T),][1:20,] %>% View


save(forest, ct_rf_train, ct_rf_test, var_rank, file = "forest.Rdata")
