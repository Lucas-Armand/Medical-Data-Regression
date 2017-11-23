#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sbn
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats.mstats import normaltest
from sklearn.neural_network import MLPRegressor
from pandas.tools.plotting import scatter_matrix
from scipy.stats import multivariate_normal
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.optimize import minimize

df = pd.read_csv('database.txt',';', decimal = '.')
columns = df.columns

# Vizualization:
#
# df.head()
# df.describe()
# len(df)
# df.hist()
# plt.show()
# scatter_matrix(df)	
# plt.show()
# correlation = df.corr()
# sbn.heatmap(correlation)
# plt.show()

# Test Vector:
trnn = [1 , 0 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]

trnn = np.array(trnn)
test = 1-trnn
test = test == 1
trnn = trnn == 1


df_test = df[test]
df_trnn = df[trnn]

# Regressão de VO2 usando Carga:

	# Gerando o dataframe para receber as variáveis da coluna '3' e suas variações  para a regressão:
df_r34 = pd.DataFrame()
df_test_r34 = pd.DataFrame()

	# Construindo resultados para polinômios de diferentes grauss
for i in range(1,9):
	df_r34['Carga Final**'+str(i)] = df_trnn[columns[2]]**i
for i in range(1,9):
	df_test_r34['Carga Final**'+str(i)] = df_test[columns[2]]**i

	# Definindo conjunto "X" e "y" para treinar o algorítimo:
Xr34 = df_r34
Yr34 = df_trnn['VO2 medido máximo (mL/kg/min)']
columns_x = Xr34.columns


	# Definindo conjunto "X" e "y" para testar o algorítimo:
xr34 = df_test_r34
yr34 = df_test['VO2 medido máximo (mL/kg/min)']


	# Construindo o vetor constante
const = pd.DataFrame({'const':np.ones(len(Xr34))},index = Xr34.index)
const2 = pd.DataFrame({'const':np.ones(len(xr34))},index = xr34.index)

for i in range(1,10):
	print i
	print ''
	results = sm.OLS(Yr34, Xr34[columns_x[0:i]].join(const)).fit()
	print results.summary()
#	print results.predict()

regr = linear_model.LinearRegression()

# resultados (1):
for i in range(1,10):
	print i
	print ''
	regr.fit(Xr34[columns_x[0:i]].join(const), Yr34)
	print 'R² Trainamento:'
	print regr.score(Xr34[columns_x[0:i]].join(const), Yr34)
	print regr.get_params()
	print 'R² Teste:'
	print regr.score(xr34[columns_x[0:i]].join(const2), yr34)
#	print results.predict()


# Prot Rsquarde train and test:
# 
# y1=[0.7750754467,0.7754524946,0.7754539306,0.7757899819,0.7759189671,0.775985176,0.7760899583]
# y2=[0.7454585045,0.7455548709,0.7454815363,0.7451629196,0.7436750232,0.7439609531,0.7434698284]
# x = range(1,8)
# fig, ax1 = plt.subplots()
# ax1.plot(x, y1, 'b-')
# ax1.set_ylabel('R^2 Treinamento', color='b')
# ax2 = ax1.twinx()
# ax2.plot(x, y2, 'r-')
# ax2.set_ylabel('R^2 Teste', color='r')
# fig.tight_layout()
# plt.show()


# Parte 2:
# Regressão de VO2 usando Carga, Peso:

	# Gerando o dataframe para receber as variáveis da coluna '3' e suas variações  para a regressão:
df_r234 = pd.DataFrame()
df_test_r234 = pd.DataFrame()

	# Construindo resultados para polinômios de diferentes grauss
for i in range(1,10):
	df_r234['Carga Final**'+str(i)] = df_trnn[columns[2]]**i
	df_r234['Peso**'+str(i)] = df_trnn[columns[1]]**i

for i in range(1,10):
	df_test_r234['Carga Final**'+str(i)] = df_test[columns[2]]**i
	df_test_r234['Peso**'+str(i)] = df_test[columns[1]]**i

	# Definindo conjunto "X" e "y" para treinar o algorítimo:
Xr234 = df_r234
Yr234 = df_trnn['VO2 medido máximo (mL/kg/min)']
columns_x = Xr234.columns


	# Definindo conjunto "X" e "y" para testar o algorítimo:
xr234 = df_test_r234
yr234 = df_test['VO2 medido máximo (mL/kg/min)']


	# Construindo o vetor constante
const = pd.DataFrame({'const':np.ones(len(Xr234))},index = Xr234.index)
const2 = pd.DataFrame({'const':np.ones(len(xr234))},index = xr234.index)

for i in range(1,3):
	print i
	print ''
	print ''
	print ''
	print Xr234[columns_x[0:2*i]].join(const).head()
	print ''
	results = sm.OLS(Yr234, Xr234[columns_x[0:2*i]].join(const)).fit()
	print results.summary()
#	print results.predict()

regr = linear_model.LinearRegression()

# Resultados (2):
for i in range(1,10):
	print i
	print ''
	regr.fit(Xr234[columns_x[0:2*i]].join(const), Yr234)
	print 'R² Trainamento:'
	print regr.score(Xr234[columns_x[0:i*2]].join(const), Yr234)
	print regr.get_params()
	print 'R² Teste:'
	print regr.score(xr234[columns_x[0:i*2]].join(const2), yr234)
#	print results.predict()

# Prot R-squarde train and test:
# 
#y1=[0.8876877002,0.899007471,0.8992172158,0.8998584384,0.9002693898,0.9002546785,0.9001756381,0.8991983449,0.8785888845]
#y2=[0.8899233697,0.9002093482,0.9007943522,0.9020746338,0.9021153518,0.9021477198,0.901987438,0.9012318944,0.8811719811]
#x = range(1,10)
#fig, ax1 = plt.subplots()
#ax1.plot(x, y1, 'b-')
#ax1.set_ylabel('R^2 Treinamento', color='b')
#ax2 = ax1.twinx()
#ax2.plot(x, y2, 'r-')
#ax2.set_ylabel('R^2 Teste', color='r')
#fig.tight_layout()
#plt.show()

# Parte 3:
# Regressão de VO2 usando Carga, Peso, Idade:

	# Gerando o dataframe para receber as variáveis da coluna '3' e suas variações  para a regressão:
df_r1234 = pd.DataFrame()
df_test_r1234 = pd.DataFrame()

	# Construindo resultados para polinômios de diferentes grauss
for i in range(1,10):
 df_r1234['Carga Final**'+str(i)] = df_trnn[columns[2]]**i
 df_r1234['Peso**'+str(i)] = df_trnn[columns[1]]**i
 df_r1234['Idade**'+str(i)] = df_trnn[columns[0]]**i


for i in range(1,10):
 df_test_r1234['Carga Final**'+str(i)] = df_test[columns[2]]**i
 df_test_r1234['Peso**'+str(i)] = df_test[columns[1]]**i
 df_test_r1234['Idade**'+str(i)] = df_test[columns[0]]**i


	# Definindo conjunto "X" e "y" para treinar o algorítimo:
Xr1234 = df_r1234
Yr1234 = df_trnn['VO2 medido máximo (mL/kg/min)']
columns_x = Xr1234.columns


	# Definindo conjunto "X" e "y" para testar o algorítimo:
xr1234 = df_test_r1234
yr1234 = df_test['VO2 medido máximo (mL/kg/min)']


	# Construindo o vetor constante
const = pd.DataFrame({'const':np.ones(len(Xr1234))},index = Xr1234.index)
const2 = pd.DataFrame({'const':np.ones(len(xr1234))},index = xr1234.index)

#for i in range(1,8):
#	print i
#	print ''
#	results = sm.OLS(Yr1234, Xr1234[columns_x[0:2*i]].join(const)).fit()
#	print results.summary()
#	print results.predict()

regr = linear_model.LinearRegression()

# Resultados (3):
for i in range(1,10):
 print i
 print ''
 regr.fit(Xr1234[columns_x[0:i*3]].join(const), Yr1234)
 print 'R² Trainamento:'
 print regr.score(Xr1234[columns_x[0:i*3]].join(const), Yr1234)
 print regr.get_params()
 print 'R² Teste:'
 print regr.score(xr1234[columns_x[0:i*3]].join(const2), yr1234)
#	print results.predict()
#y1=[0.8913319529,0.9004000064,0.9020204391,0.9030626757,0.9030755474,0.9030846293,0.903019415,0.9025699331,0.8882040509]
#y2=[0.8899630108,0.9000797872,0.9008599332,0.9016978773,0.9018695847,0.9017418306,0.9018294547,0.9013785525,0.8841431482]
#
#x = range(1,len(y1)+1)
#fig, ax1 = plt.subplots()
#ax1.plot(x, y1, 'b-')
#ax1.set_ylabel('R^2 Treinamento', color='b')
#ax2 = ax1.twinx()
#ax2.plot(x, y2, 'r-')
#ax2.set_ylabel('R^2 Teste', color='r')
#fig.tight_layout()
#plt.show()

# Questão 2:

# Parte 1
dfg34_trnn = df_trnn[['Carga Final','VO2 medido máximo (mL/kg/min)']]
dfg34_test = df_test[['Carga Final','VO2 medido máximo (mL/kg/min)']]
mean = dfg34_trnn.mean()
covarr = dfg34_trnn.cov()

x, y = np.mgrid[0:500:1, 0:80:1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal(mean,cov=covarr)
plt.contourf(x, y, rv.pdf(pos))

col = dfg34_trnn.columns
plt.scatter(dfg34_trnn[col[0]],list(dfg34_trnn[col[1]]),'b')
plt.scatter(dfg34_test[col[0]],list(dfg34_test[col[1]]),'r')

plt.show()


# Parte 2

dfg234_trnn = df_trnn[['Peso (kg)', 'Carga Final', 'VO2 medido máximo (mL/kg/min)']]
dfg234_test = df_test[['Peso (kg)', 'Carga Final', 'VO2 medido máximo (mL/kg/min)']]
mean234 = dfg234_trnn.mean()
covarr234 = dfg234_trnn.cov()
rv = multivariate_normal(mean234,cov=covarr234)
z = np.mgrid[0:80:1]
pos = np.empty(z.shape+(3,))
pos [:,2] = z

x = [60,80,100]
y = [100,200,300]
for i in range(3):
	pos[:,0]=x[i];pos[:,1]=y[i]
	P_z = rv.pdf(pos)
	norm = np.trapz(P_z)
	P_z_cond = P_z/norm
	#plt.plot(P_z_cond)
	mean = sum(P_z_cond*z)
	print mean
	interval = [mean-5,mean+5]
	delt = np.mgrid[interval[0]:interval[1]:1]
	points = np.empty(delt.shape+(3,))
	points[:,2] = delt
	points[:,0]=x[i];points[:,1]=y[i]
	print points
	P_z_cond = rv.pdf(points)/norm
	prob = np.trapz(P_z_cond)
	print prob
	print ''

# Parte 3

ind = dfg234_test.index
n = len(ind)
y_pred = []
for i in range(n):
	point = dfg234_test.loc[ind[i]][:2]
	res = minimize(lambda x: -rv.pdf([point[0],point[1],x]),50,method='powell')
	mean = res.x
	print point[0],point[1],mean
	y_pred.append(mean)
y_true = dfg234_test[col[1]]
print ' Resultado do modelo gaussiano:'
print r2_score(y_true, y_pred)



# Questão 3:

# Parte 1

df_test_18_40 = df_test[df_test['IDADE (anos)']<40]
df_test_40_all = df_test[df_test['IDADE (anos)']>=40]
df_test_40_inf = df_test[df_test['IDADE (anos)']>=40]
df_test_40_65 = df_test_40_inf[df_test_40_inf['IDADE (anos)']<65]
df_test_65_inf = df_test_40_inf[df_test_40_inf['IDADE (anos)']>=65]


df_trnn_18_40 = df_trnn[df_trnn['IDADE (anos)']<40]
df_trnn_40_all = df_trnn[df_trnn['IDADE (anos)']>=40]
df_trnn_40_inf = df_trnn[df_trnn['IDADE (anos)']>=40]
df_trnn_40_65 = df_trnn_40_inf[df_trnn_40_inf['IDADE (anos)']<65]
df_trnn_65_inf = df_trnn_40_inf[df_trnn_40_inf['IDADE (anos)']>=65]


