
# 1 - separar bem as aguas
x <- df_temp[,-1]
y <- df_temp$PATOLOGIA

# 2 - passar as variaveis categorias a one hot-encoding
num <-
  x[index,] %>% select_if(~!is.numeric(.x)) %>%
  caret::dummyVars("~.", data = .) %>%
  predict(object = .,
          newdata = x[index,] %>% select_if(~!is.numeric(.x)))

cat <-
  # 3 - normalizar as variaveis numericas, separando teste de validação para não contaminar
  x[index,] %>% select_if(is.numeric) %>%
  preProcess(method = "range") %>%
  predict(object = .,
          newdata = x[index,] %>% select_if(is.numeric))

num_val <-
  x[-index,] %>% select_if(~!is.numeric(.x)) %>%
  caret::dummyVars("~.", data = .) %>%
  predict(object = .,
          newdata = x[-index,] %>% select_if(~!is.numeric(.x)))

cat_val <-
  # 3 - normalizar as variaveis numericas
  x[-index,] %>% select_if(is.numeric) %>%
  preProcess(method = "range") %>%
  predict(object = .,
          newdata = x[-index,] %>% select_if(is.numeric))

x_norm <-cbind(num,cat) %>% as.matrix
x_val <- cbind(num_val, cat_val) %>% as.matrix

# 4 - binarizar (palavra linda) a variavel resposta
y_norm <- ifelse(y == "Normal",1,0) 
# %>% as.numeric() %>% as.matrix

# 5 - preparar a rede
model_nn <- keras_model_sequential()

model_nn %>%
  layer_dense(input_shape = dim(x_norm)[2],units=20,name="H1",use_bias=T, activation = 'relu') %>%
  layer_dense(units = 20, use_bias =  T, activation = 'relu') %>%
  layer_dense(units = 1,name="Output") 

# loss, optimizer, metrics
model_nn %>% keras::compile(loss = 'binary_crossentropy', 
                            optimizer = optimizer_rmsprop(lr = 0.0001),
                            metrics = c('accuracy'))

# 6 Treinar
history <- model_nn %>% fit(
  x_norm, y_norm[index], 
  epochs = 30, batch_size = 128, 
  validation_data = list(x_val,y_norm[-index]))


# # 7 Olhar
# plot(history)
# 
# 
# sum((model %>% predict(inp)))/sum(y_nn)
# 
# # tensorboard(action = "stop")
# 
# plot(history) + theme_light()
# 
# data.frame(p.GUU=y_nn,
#            p.GUU_hat=(model %>% predict(inp)),
#            grupo=train[,"EESPECIE"]) %>%
#   # filter(lota %in% c("SINES","OLHAO","VRSA")) %>%
#   ggplot+
#   geom_point(aes(x=p.GUU,y=p.GUU_hat,color=EESPECIE))+
#   geom_abline(slope=1,intercept=0) +
#   labs(x = "Quantidade real de GUU (kg)", y = "quantidade prevista de GUU (kg)", col = "") + 
#   theme_light()
# 