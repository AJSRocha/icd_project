grelha <- expand.grid(mtry = c(1:16))

forest <- caret::train(modelo,
                       method="rf",
                       data=df_temp[index,],
                       trControl=control,
                       tuneGrid=grelha,
                       preProc = c("center","scale"),
                       ntree=500,
                       metric="Accuracy")

summary(forest)
print(forest)

# confusion matrix of the training data
caret::confusionMatrix(forest)

# confusion matrix of the holdout data
confusionMatrix(df_num[-index,]$PATOLOGIA,
                predict(forest, newdata = df_num[-index,]))

var_rank<-data.frame(variables=rownames(forest$finalModel$importance),importance=forest$finalModel$importance)

var_rank[order(var_rank$IncNodePurity,decreasing=T),][1:20,] %>% View