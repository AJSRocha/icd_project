# Data cleaning
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import math
import seaborn as sns
import sklearn as skl
import scikitplot as skplt
import warnings
import matplotlib.pyplot as plt
import csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import statsmodels.api as sm

df = pd.read_excel('UCMF.xls')

weight_0_36 = pd.read_excel('wtageinf.xls')
height_0_36 = pd.read_excel('lenageinf.xls')
weight_2_20 = pd.read_excel('wtage.xls')
height_2_20 = pd.read_excel('statage.xls')
bmi_2_20 = pd.read_excel('bmiagerev.xls')

df = df.drop(columns='ID') #remoção da coluna ID
df = df[df['IDADE'].notna()] #remoção de NaN de idade
df = df[df['IDADE']!='#VALUE!'] #remoção de #VALUE de idade. Nestes casos ou a data de atendimento ou de nascença é inválida.
df['IDADE']=df['IDADE'].astype(float)
df = df[df['IDADE']>0.0] #remoção dos valores negativos de idade e maiores que 20 anos
df = df[df['IDADE']<20.0]
df = df[df['Atendimento'].notna()] #remoção de NaN de atendimento. assume-se que não foram atendidos. Não é ncessário para o problema.
df = df.drop(columns='Atendimento') #remoção da coluna atendimento
df = df[df['DN'].notna()] #remoção de NaN de data de nascimento. Não é ncessário para o problema.
df = df.drop(columns='DN') #remoção da coluna DN
df = df.sample(frac=1).reset_index(drop=True)

# Tratamento SEXO
df['SEXO'] = df['SEXO'].fillna('Indeterminado')
df['SEXO'] = df['SEXO'].replace({'Masculino':'M', 'Feminino':'F', 'masculino':'M'})
df = df[(df['SEXO']=='M') | (df['SEXO']=='F')] #remoção dos indeterminados

# Tratamento Peso, Altura e IMC
m_weight_0_36 = weight_0_36[(weight_0_36['Sex']==1) & (weight_0_36['Agemos']!=0.0)]
f_weight_0_36 = weight_0_36[(weight_0_36['Sex']==2) & (weight_0_36['Agemos']!=0.0)]
m_height_0_36 = height_0_36[(height_0_36['Sex']==1) & (weight_0_36['Agemos']!=0.0)]
f_height_0_36 = height_0_36[(height_0_36['Sex']==2) & (weight_0_36['Agemos']!=0.0)]

m_weight_2_20 = weight_2_20[(weight_2_20['Sex']==1) & (weight_2_20['Agemos']!=0.0)]
f_weight_2_20 = weight_2_20[(weight_2_20['Sex']==2) & (weight_2_20['Agemos']!=0.0)]
m_height_2_20 = height_2_20[(height_2_20['Sex']==1) & (weight_2_20['Agemos']!=0.0)]
f_height_2_20 = height_2_20[(height_2_20['Sex']==2) & (weight_2_20['Agemos']!=0.0)]

m_bmi_2_20 = bmi_2_20[(bmi_2_20['Sex']==1) & (bmi_2_20['Agemos']!=0.0)]
f_bmi_2_20 = bmi_2_20[(bmi_2_20['Sex']==2) & (bmi_2_20['Agemos']!=0.0)]

m_weight_0_36 = m_weight_0_36[m_weight_0_36['Agemos'].notna()]
f_weight_0_36 = f_weight_0_36[f_weight_0_36['Agemos'].notna()]
m_height_0_36 = m_height_0_36[m_height_0_36['Agemos'].notna()]
f_height_0_36 = f_height_0_36[f_height_0_36['Agemos'].notna()]

m_weight_2_20 = m_weight_2_20[m_weight_2_20['Agemos'].notna()]
f_weight_2_20 = f_weight_2_20[f_weight_2_20['Agemos'].notna()]
m_height_2_20 = m_height_2_20[m_height_2_20['Agemos'].notna()]
f_height_2_20 = f_height_2_20[f_height_2_20['Agemos'].notna()]


m_bmi_2_20 = m_bmi_2_20[m_bmi_2_20['Agemos'].notna()]
f_bmi_2_20 = f_bmi_2_20[f_bmi_2_20['Agemos'].notna()]

m_weight_0_36.loc[:,'Agemos'] = m_weight_0_36['Agemos'].astype(float)
f_weight_0_36.loc[:,'Agemos'] = f_weight_0_36['Agemos'].astype(float)
m_height_0_36.loc[:,'Agemos'] = m_height_0_36['Agemos'].astype(float)
f_height_0_36.loc[:,'Agemos'] = f_height_0_36['Agemos'].astype(float)

m_weight_2_20.loc[:,'Agemos'] = m_weight_2_20['Agemos'].astype(float)
f_weight_2_20.loc[:,'Agemos'] = f_weight_2_20['Agemos'].astype(float)
m_height_2_20.loc[:,'Agemos'] = m_height_2_20['Agemos'].astype(float)
f_height_2_20.loc[:,'Agemos'] = f_height_2_20['Agemos'].astype(float)

m_bmi_2_20.loc[:,'Agemos'] = m_bmi_2_20['Agemos'].astype(float)
f_bmi_2_20.loc[:,'Agemos'] = f_bmi_2_20['Agemos'].astype(float)

m_weight_0_36.loc[:,'P50'] = m_weight_0_36['P50'].astype(float)
f_weight_0_36.loc[:,'P50'] = f_weight_0_36['P50'].astype(float)
m_height_0_36.loc[:,'P50'] = m_height_0_36['P50'].astype(float)
f_height_0_36.loc[:,'P50'] = f_height_0_36['P50'].astype(float)

m_weight_2_20.loc[:,'P50'] = m_weight_2_20['P50'].astype(float)
f_weight_2_20.loc[:,'P50'] = f_weight_2_20['P50'].astype(float)
m_height_2_20.loc[:,'P50'] = m_height_2_20['P50'].astype(float)
f_height_2_20.loc[:,'P50'] = f_height_2_20['P50'].astype(float)

m_bmi_2_20.loc[:,'P50'] = m_bmi_2_20['P50'].astype(float)
f_bmi_2_20.loc[:,'P50'] = f_bmi_2_20['P50'].astype(float)

m_weight_0_36.loc[:,'Agemos'] = m_weight_0_36['Agemos'].div(12.0)
f_weight_0_36.loc[:,'Agemos'] = f_weight_0_36['Agemos'].div(12.0)
m_height_0_36.loc[:,'Agemos'] = m_height_0_36['Agemos'].div(12.0)
f_height_0_36.loc[:,'Agemos'] = f_height_0_36['Agemos'].div(12.0)

m_weight_2_20.loc[:,'Agemos'] = m_weight_2_20['Agemos'].div(12.0)
f_weight_2_20.loc[:,'Agemos'] = f_weight_2_20['Agemos'].div(12.0)
m_height_2_20.loc[:,'Agemos'] = m_height_2_20['Agemos'].div(12.0)
f_height_2_20.loc[:,'Agemos'] = f_height_2_20['Agemos'].div(12.0)

m_bmi_2_20.loc[:,'Agemos'] = m_bmi_2_20['Agemos'].div(12.0)
f_bmi_2_20.loc[:,'Agemos'] = f_bmi_2_20['Agemos'].div(12.0)

f_peso_idade1_curva = np.polyfit(f_weight_0_36['Agemos'], f_weight_0_36['P50'],2)
f_altr_idade1_curva = np.polyfit(f_height_0_36['Agemos'], f_height_0_36['P50'],2)
m_peso_idade1_curva = np.polyfit(m_weight_0_36['Agemos'], m_weight_0_36['P50'],2)
m_altr_idade1_curva = np.polyfit(m_height_0_36['Agemos'], m_height_0_36['P50'],2)

f_peso_idade2_curva = np.polyfit(f_weight_2_20['Agemos'], f_weight_2_20['P50'],2)
f_altr_idade2_curva = np.polyfit(f_height_2_20['Agemos'], f_height_2_20['P50'],2)
m_peso_idade2_curva = np.polyfit(m_weight_2_20['Agemos'], m_weight_2_20['P50'],2)
m_altr_idade2_curva = np.polyfit(m_height_2_20['Agemos'], m_height_2_20['P50'],2)

m_bmi_2_20_curva = np.polyfit(m_bmi_2_20['Agemos'], m_bmi_2_20['P50'], 3)
f_bmi_2_20_curva = np.polyfit(f_bmi_2_20['Agemos'], f_bmi_2_20['P50'], 3)

predict_f_peso_idade1 = np.poly1d(f_peso_idade1_curva)
predict_f_altr_idade1 = np.poly1d(f_altr_idade1_curva)
predict_m_peso_idade1 = np.poly1d(m_peso_idade1_curva)
predict_m_altr_idade1 = np.poly1d(m_altr_idade1_curva)

predict_f_peso_idade2 = np.poly1d(f_peso_idade2_curva)
predict_f_altr_idade2 = np.poly1d(f_altr_idade2_curva)
predict_m_peso_idade2 = np.poly1d(m_peso_idade2_curva)
predict_m_altr_idade2 = np.poly1d(m_altr_idade2_curva)

predict_m_bmi_2_20 = np.poly1d(m_bmi_2_20_curva)
predict_f_bmi_2_20 = np.poly1d(f_bmi_2_20_curva)

df1 = df['IDADE'][(df.loc[:,'Peso']<=0.0) & (df.loc[:,'IDADE'] < 3.0) & (df.loc[:,'SEXO'] == 'F')]
df['Peso'][(df.loc[:,'Peso']<=0.0) & (df.loc[:,'IDADE'] < 3.0) & (df.loc[:,'SEXO'] == 'F')] = predict_f_peso_idade1(df1)

df2 = df['IDADE'][(df.loc[:,'Peso']<=0.0) & (df.loc[:,'IDADE'] < 3.0) & (df.loc[:,'SEXO'] == 'M')]
df['Peso'][(df.loc[:,'Peso']<=0.0) & (df.loc[:,'IDADE'] < 3.0) & (df.loc[:,'SEXO'] == 'M')] = predict_m_peso_idade1(df2)

df3 = df['IDADE'][(df.loc[:,'Peso']<=0.0) & (df.loc[:,'IDADE'] >= 3.0) & (df.loc[:,'SEXO'] == 'F')]
df['Peso'][(df.loc[:,'Peso']<=0.0) & (df.loc[:,'IDADE'] >= 3.0) & (df.loc[:,'SEXO'] == 'F')] = predict_f_peso_idade2(df3)

df4 = df['IDADE'][(df.loc[:,'Peso']<=0.0) & (df.loc[:,'IDADE'] >= 3.0) & (df.loc[:,'SEXO'] == 'M')]
df['Peso'][(df.loc[:,'Peso']<=0.0) & (df.loc[:,'IDADE'] >= 3.0) & (df.loc[:,'SEXO'] == 'M')] = predict_m_peso_idade2(df4)

df5 = df['IDADE'][(df.loc[:,'Altura']<=0) & (df.loc[:,'IDADE'] < 3.0) & (df.loc[:,'SEXO'] == 'F')]
df['Altura'][(df.loc[:,'Altura']<=0) & (df.loc[:,'IDADE'] < 3.0) & (df.loc[:,'SEXO'] == 'F')] = predict_f_altr_idade1(df5)

df6 = df['IDADE'][(df.loc[:,'Altura']<=0) & (df.loc[:,'IDADE'] < 3.0) & (df.loc[:,'SEXO'] == 'M')]
df['Altura'][(df.loc[:,'Altura']<=0) & (df.loc[:,'IDADE'] < 3.0) & (df.loc[:,'SEXO'] == 'M')] = predict_m_altr_idade1(df6)

df7 = df['IDADE'][(df.loc[:,'Altura']<=0) & (df.loc[:,'IDADE'] >= 3.0) & (df.loc[:,'SEXO'] == 'F')]
df['Altura'][(df.loc[:,'Altura']<=0) & (df.loc[:,'IDADE'] >= 3.0) & (df.loc[:,'SEXO'] == 'F')] = predict_f_altr_idade2(df7)

df8 = df['IDADE'][(df.loc[:,'Altura']<=0) & (df.loc[:,'IDADE'] >= 3.0) & (df.loc[:,'SEXO'] == 'M')]
df['Altura'][(df.loc[:,'Altura']<=0) & (df.loc[:,'IDADE'] >= 3.0) & (df.loc[:,'SEXO'] == 'M')] = predict_m_altr_idade2(df8)

def outls_w(data_m, data_f, min_age, max_age, tol):

    data_m['Agemos'] = round(data_m['Agemos'], 4)
    data_f['Agemos'] = round(data_f['Agemos'], 4)
    df.reset_index(drop=True, inplace=True)

    count = 0
    rows_to_drop = []
    for q in range(df.shape[0]):
            if((df['IDADE'].iloc[q] >= min_age) & (df['IDADE'].iloc[q] < max_age) & (df['SEXO'].iloc[q] == 'M')):
                # encontrar o p97 do valor mais proximo da idade na tabela de imputacao
                p97  = round (float (data_m.loc[(data_m['Agemos']==(min(data_m['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P97']), 3)
                p3   = round (float (data_m.loc[(data_m['Agemos']==(min(data_m['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P3']), 3)

                if((df['Peso'].iloc[q]) > (p97 *(1+tol/100))):
                    rows_to_drop.append (q)
                    count += 1
                    #print (df['IDADE'].iloc[q], df['Peso'].iloc[q] )

                elif((df['Peso'].iloc[q]) < (p3 *(1-tol/100))):
                    rows_to_drop.append (q)
                    count += 1
                    
            elif((df['IDADE'].iloc[q] >= min_age) & (df['IDADE'].iloc[q] < max_age) & (df['SEXO'].iloc[q] == 'F')):
                # encontrar o p97 do valor mais proximo da idade na tabela de imputacao
                p97  = round (float (data_f.loc[(data_f['Agemos']==(min(data_f['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P97']), 3)
                p3   = round (float (data_f.loc[(data_f['Agemos']==(min(data_f['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P3']), 3)

                if((df['Peso'].iloc[q]) > (p97 *(1+tol/100))):
                    rows_to_drop.append (q)
                    count += 1
                    #print (df['IDADE'].iloc[q], df['Peso'].iloc[q] )

                elif((df['Peso'].iloc[q]) < (p3 *(1-tol/100))):
                    rows_to_drop.append (q)
                    count += 1
    print (count)
    print('ok')
    return rows_to_drop;
  
def outls_h(data_m, data_f, min_age, max_age, tol):
    data_f['Agemos'] = round(data_f['Agemos'], 4)
    df.reset_index(drop=True, inplace=True)

    count = 0
    rows_to_drop = []
    for q in range(df.shape[0]):
            if((df['IDADE'].iloc[q] >= min_age) & (df['IDADE'].iloc[q] < max_age) & (df['SEXO'].iloc[q] == 'M')):
                # encontrar o p97 do valor mais proximo da idade na tabela de imputacao
                p97  = round (float (data_m.loc[(data_m['Agemos']==(min(data_m['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P97']), 3)
                p3   = round (float (data_m.loc[(data_m['Agemos']==(min(data_m['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P3']), 3)

                if((df['Altura'].iloc[q]) > (p97 *(1+tol/100))):
                    rows_to_drop.append (q)
                    count += 1
                    #print (df['IDADE'].iloc[q], df['Peso'].iloc[q] )

                elif((df['Altura'].iloc[q]) < (p3 *(1-tol/100))):
                    rows_to_drop.append (q)
                    count += 1
                    
            elif((df['IDADE'].iloc[q] >= min_age) & (df['IDADE'].iloc[q] < max_age) & (df['SEXO'].iloc[q] == 'F')):
                # encontrar o p97 do valor mais proximo da idade na tabela de imputacao
                p97  = round (float (data_f.loc[(data_f['Agemos']==(min(data_f['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P97']), 3)
                p3   = round (float (data_f.loc[(data_f['Agemos']==(min(data_f['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P3']), 3)

                if((df['Altura'].iloc[q]) > (p97 *(1+tol/100))):
                    rows_to_drop.append (q)
                    count += 1
                    #print (df['IDADE'].iloc[q], df['Peso'].iloc[q] )

                elif((df['Altura'].iloc[q]) < (p3 *(1-tol/100))):
                    rows_to_drop.append (q)
                    count += 1
    print (count)
    print('ok')
    return rows_to_drop;
  
rows = outls_w(m_weight_0_36, f_weight_0_36, 0, 3, 20)
df = df.drop(rows)
df = df.sample(frac=1).reset_index(drop=True)

rows = outls_h(m_height_0_36, f_height_0_36, 0, 3, 20)
df = df.drop(rows)
df = df.sample(frac=1).reset_index(drop=True)

rows = outls_w(m_weight_2_20, f_weight_2_20, 3, 20, 20)
df = df.drop(rows)
df = df.sample(frac=1).reset_index(drop=True)

rows = outls_h(m_height_2_20, f_height_2_20, 3, 20, 20)
df = df.drop(rows)
df = df.sample(frac=1).reset_index(drop=True)

df['IMC'][(df.loc[:,'SEXO'] == 'F')] = predict_f_bmi_2_20(df['IDADE'])

df['IMC'][(df.loc[:,'SEXO'] == 'M')] = predict_m_bmi_2_20(df['IDADE'])

df.loc[:,'IMC'] = df['Peso']/((df['Altura']/100)**2)

df['IDADE']=df['IDADE'].astype(int) #Conversão da idade para inteiros.

df = df[df['IDADE']>0]

df = df[df['IMC']<50]

df = df.drop(columns='Convenio') #Não é necessário ao problema
df = df.drop(columns='PULSOS') #Não é necessário ao problema
df['NORMAL X ANORMAL'] = df['NORMAL X ANORMAL'].str.replace('anormal','Anormal')
df['NORMAL X ANORMAL'] = df['NORMAL X ANORMAL'].str.replace('Normais','Normal')
df['NORMAL X ANORMAL'] = df['NORMAL X ANORMAL'].str.replace('anormal','Anormal')
df['NORMAL X ANORMAL'] = df['NORMAL X ANORMAL'].str.replace('Normais','Normal')
# df['SOPRO'].value_counts()
df['SOPRO'] = df['SOPRO'].replace({'sistólico':'Sistólico', 'contínuo':'Contínuo'})
df['SOPRO'] = df['SOPRO'].replace({'Sistólico':'presente', 'Contínuo':'presente', 'diastólico':'presente', 'Sistolico e diastólico':'presente'})
df['SOPRO'].value_counts()

# Tratamento FC
def testKNN_r(df_prev, df_obj, std):
    x = df_prev
    y = df_obj
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    from sklearn.preprocessing import normalize
    
    if std == 1:
        x = sc.fit_transform(x)
    else:
        x = normalize(x)
    
    errors = []
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)
    k_ok = 0
    
    gain = 0
    regressor = KNeighborsRegressor(n_neighbors = 1)
    regressor.fit(X_train,Y_train)
    error_0 = round(mean_absolute_error(Y_test,regressor.predict(X_test)),3)
    
    for k in range (1,20,2):
        regressor = KNeighborsRegressor(n_neighbors = k)
        regressor.fit(X_train,Y_train)
        error = round(mean_absolute_error(Y_test,regressor.predict(X_test)),3)
        gain = round(error_0 - error,3)
        error_0 = error
        errors.append({'K':k, 'mae':error, 'gain':gain})
    
    print(errors)
    
# df.FC.unique()
import re
for i in range(df.shape[0]):
    if type(df['FC'].iloc[i]) == str:
        v = re.findall(r'\d+', df['FC'].iloc[i]) #retorna um vector com os valores númericos encontrados. Ex: 120-140 --> 120;140
        v = [int(k) for k in v]
        df['FC'].iloc[i] = int(abs(v[1]-v[0])/2) #calcula o valor intermédio entre os dois valores e guarda

df = df.sample(frac=1).reset_index(drop=True)

df2 = df
df2 = df2[df2['FC'].notna()]
df2 = df2.sample(frac=1).reset_index(drop=True)
df2['FC'] = df2['FC'].astype(int)

# df2

df = df[df['FC'] <= 230] #Valor máximo de frequência cardíaca
df = df.sample(frac=1).reset_index(drop=True)

X = df2[['IMC','IDADE']]
Y = df2['FC']

# testKNN_r(X,Y,1)
# testKNN_r(X,Y,0)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors = 9)
regressor.fit(X_train, Y_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

mse = mean_squared_error(Y_test, regressor.predict(X_test))
mae = mean_absolute_error(Y_test, regressor.predict(X_test))

# print(mse, mae)

for k in range(df.shape[0]): #em todo as linhas
    if pd.isnull(df['FC'].iloc[k]):   #caso um valor de FC seja NaN
        Q = df[['IMC','IDADE']].iloc[k]
        df['FC'].iloc[k] = regressor.predict(Q)
        
df = df[df['FC'] >= 40]  #Valor mínimo de frequência cardiaca

df['HDA 1'] = df['HDA 1'].fillna('Sem histórico')
# df['HDA 1'].value_counts()
# df['HDA 1'].unique()

df['HDA2'] = df['HDA2'].fillna('Sem histórico')
# df['HDA2'].value_counts()
# df['HDA2'].unique()
df[df['PA SISTOLICA']<500]
df = df.sample(frac=1).reset_index(drop=True)

df3 = df
df3 = df3[(df3['PA SISTOLICA']<500) & (df3['PA SISTOLICA']>60) & (df3['PA DIASTOLICA']>20)]
df3 = df3[(df3['PA SISTOLICA'].notna()) & (df3['PA DIASTOLICA'].notna())]
df3 = df3.sample(frac=1).reset_index(drop=True)

Z = df3[['Peso','Altura','IDADE']]
W = df3[['PA SISTOLICA']]
# testKNN_r(Z,W,1)
# testKNN_r(Z,W,0)

from sklearn.model_selection import train_test_split
Z_train, Z_test, W_train, W_test = train_test_split(Z,W,test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Z_train = sc.fit_transform(Z_train)
Z_test = sc.transform(Z_test)

from sklearn.neighbors import KNeighborsRegressor
regressor2 = KNeighborsRegressor(n_neighbors = 7)
regressor2.fit(Z_train, W_train)

mse2 = mean_squared_error(W_test, regressor2.predict(Z_test))
mae2 = mean_absolute_error(W_test, regressor2.predict(Z_test))

# print(mse2, mae2)

E = df3[['Peso','Altura','IDADE']]
R = df3[['PA DIASTOLICA']]
# testKNN_r(E,R,1)
# testKNN_r(E,R,0)

E_train, E_test, R_train, R_test = train_test_split(E,R,test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
E_train = sc.fit_transform(E_train)
E_test = sc.transform(E_test)

from sklearn.neighbors import KNeighborsRegressor
regressor3 = KNeighborsRegressor(n_neighbors = 7)
regressor3.fit(E_train, R_train)

mse3 = mean_squared_error(R_test, regressor3.predict(E_test))
mae3 = mean_absolute_error(R_test, regressor3.predict(E_test))

# print(mse3, mae3)

for o in range(df.shape[0]): #em todo as linhas
    if pd.isnull((df['PA SISTOLICA'].iloc[o])):
        K = df[['Peso','Altura','IDADE']].iloc[o].values.reshape(1,-1)
        #K = K.values.reshape(1,-1)
        df['PA SISTOLICA'].iloc[o] = regressor2.predict(K)
    
    if pd.isnull((df['PA DIASTOLICA'].iloc[o])):
        Q = df[['Peso','Altura','IDADE']].iloc[o].values.reshape(1,-1)
        #Q = Q.values.reshape(1,-1)
        df['PA DIASTOLICA'].iloc[o] = regressor3.predict(Q)

df = df[(df['PA SISTOLICA']<200) & (df['PA SISTOLICA']>60) & (df['PA DIASTOLICA']>20)]

# df.shape[0]
df = df[(df['B2'].notna()) & (df['NORMAL X ANORMAL'].notna())]
df = df.drop(columns='PPA') 
# df['HDA 1'].value_counts()
# df.isna().sum()

dfx = df[df['HDA 1']== "Ganho de peso"]

df['HDA 1'][(df['MOTIVO2']=='6 - Dor precordial') & (df['HDA 1']=='Sem histórico')]='Dor precordial'
df['HDA 1'][(df['MOTIVO2']=='6 - Cianose') & (df['HDA 1']== 'Sem histórico')] = 'Cianose'
df.loc[(((df['HDA 1'].isna())) & (df['MOTIVO2']=='6 - Cianose e dispnéia')), 'HDA 1'] = "Cianose"
df.loc[(((df['HDA2'].isna())) & (df['MOTIVO2']=='6 - Cianose e dispnéia')), 'HDA2'] = "Dispneia"

def over_imc(data_m, data_f):
    data_f['Agemos'] = round(data_f['Agemos'], 4)
    df.reset_index(drop=True, inplace=True)

   
    for q in range(df.shape[0]):
            if((df['SEXO'].iloc[q] == 'M') & (df['HDA 1'].iloc[q] == 'Sem histórico')):
                # encontrar o p97 do valor mais proximo da idade na tabela de imputacao
                p85  = round (float (data_m.loc[(data_m['Agemos']==(min(data_m['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P85']), 3)
                #p3   = round (float (data_m.loc[(data_m['Agemos']==(min(data_m['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P3']), 3)

                if((df['IMC'].iloc[q]) > p85):
                    df['HDA 1'].iloc[q] = 'Ganho de peso'
                    #print (df['IDADE'].iloc[q], df['Peso'].iloc[q] )
                    #print('M added')
                                    
            elif((df['SEXO'].iloc[q] == 'F') & (df['HDA 1'].iloc[q] == 'Sem histórico')):
                # encontrar o p97 do valor mais proximo da idade na tabela de imputacao
                p85  = round (float (data_f.loc[(data_f['Agemos']==(min(data_f['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P85']), 3)
                #p3   = round (float (data_f.loc[(data_f['Agemos']==(min(data_f['Agemos'], key=lambda x:abs(x-df['IDADE'].iloc [q]))))]['P3']), 3)

                if((df['IMC'].iloc[q]) > p85):
                    df['HDA 1'].iloc[q] = 'Ganho de peso'
                    #print (df['IDADE'].iloc[q], df['Peso'].iloc[q] )
                    #print('F added')

over_imc(m_bmi_2_20, f_bmi_2_20)
df = df.drop(columns = 'HDA2')
df.rename(columns={'HDA 1':'HDA'},inplace=True)
df.insert(13, "MOTIVO", np.nan)

for o in range(df.shape[0]):
    if ((str(df['MOTIVO2'].iloc[o])!= 'nan') & (str(df['MOTIVO2'].iloc[o])!= 'Outro')):
        if len(df[(df['MOTIVO1'] ==df['MOTIVO1'].iloc[o]) & (df['MOTIVO2'] == df['MOTIVO2'].iloc[o])]) >=90:
            df['MOTIVO'].iloc[o] = str(df['MOTIVO1'].iloc[o]) + ' (' + str(df['MOTIVO2'].iloc[o]) + ')'
        else:
            df['MOTIVO'].iloc[o] = str(df['MOTIVO1'].iloc[o])
    elif((str(df['MOTIVO1'].iloc[o]) == 'nan') | pd.isnull(df['MOTIVO1'].iloc[o])):
        df['MOTIVO'].iloc[o] = '7 - Outro'
    else:
        df['MOTIVO'].iloc[o] = str(df['MOTIVO1'].iloc[o])
        
df = df.drop(columns='MOTIVO1')
df = df.drop(columns='MOTIVO2') 


#Reescreve o csv, cuidado!
# df.to_csv('df_final.csv', index=False)
