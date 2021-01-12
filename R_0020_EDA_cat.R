

## PROPOSTA DE MODELO

df %>%
  filter(PA_SISTOLICA < 500) %>%
  ggplot +
  geom_point(aes(x = PA_SISTOLICA, y = PA_DIASTOLICA, col = PATOLOGIA)) +
  theme_light()


# Patologia associada a sistolico -> passar a presente/ausente
table(df$SOPRO, df$PATOLOGIA) %>%
  data.frame() %>%
  ggplot +
  geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90))

# B2 não fornece grande distinção em termos de patologia - entre normal e anormal pode ter info
table(df$B2, df$PATOLOGIA) %>%
  data.frame() %>%
  ggplot +
  geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90))

# PPA também não
table(df$PPA, df$PATOLOGIA) %>%
  data.frame() %>%
  ggplot +
  geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90))

# Alguma distinção entre sexo
table(df$SEXO, df$PATOLOGIA) %>%
  data.frame() %>%
  ggplot +
  geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90))

# Idade é um factor entre patologia
df %>%
  mutate(IDADE_2 = trunc(IDADE)) %>%
  filter(IDADE_2 >= 0) %>%
  select(IDADE_2,PATOLOGIA) %>%
  table() %>%
  data.frame() %>%
  ggplot +
  geom_tile(aes(x = IDADE_2, y = PATOLOGIA, fill = Freq)) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90))

# Alguma distinção para alguns motivos
table(df$MOTIVO1, df$PATOLOGIA) %>%
  data.frame() %>%
  ggplot +
  geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90))

# Alguma distinção para alguns motivos
table(df$MOTIVO2, df$PATOLOGIA) %>%
  data.frame() %>%
  ggplot +
  geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90))

# Alguma distinção para alguns motivos
table(df$HDA1, df$PATOLOGIA) %>%
  data.frame() %>%
  ggplot +
  geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90))

table(df$HDA2, df$PATOLOGIA) %>%
  data.frame() %>%
  ggplot +
  geom_tile(aes(x = Var1, y = Var2, fill = Freq)) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90))


