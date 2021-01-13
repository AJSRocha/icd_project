
# grelha_svm <- expand.grid(tau = c(0.01))

control <- caret::trainControl(method = "cv", number = 10)

svm_l <- caret::train(modelo_num,
                      method = 'svmPoly',
                      data=df_num[index,],
                      trControl=control,
                      # tuneGrid=grelha_svm,
                      # preProc = c("center","scale"),
                      metric="Accuracy")

svm_l
confusionMatrix(svm_l)