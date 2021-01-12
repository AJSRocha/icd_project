

### Imputar sexo com base em gaussian naive bayes

library(naivebayes)

# with df

names(df)
summary(factor(df$SEXO))


x <- df[df$SEXO != "I",c("Peso","Altura","IMC","IDADE")]
y <- df[df$SEXO != "I",]$SEXO %>% factor

sex_class <- gaussian_naive_bayes(x,y)

y_pred <- predict(sex_class) 
table(y)

caret::confusionMatrix(y_pred,na.omit(y))


