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

df = pd.read_csv('database.txt','\t', decimal = '.')
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
test = [1 , 0 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]



df = df.drop('COI',1)	
df = df.drop('V0003_p.1',1)
df['diff_REC'] = df['REC_N+1']-df['REC_N']
df['diff_COI'] = df['COI_N+1']-df['COI_N']
df['diff_VBPI'] = df['VBPI_N+1']-df['VBPI_N']
columns = df.columns

# Para eliminar nan e inf:
df = df[np.invert(np.isnan(df['diff_COI']))]
df = df[np.invert(np.isinf(df['diff_COI']))]
df = df[np.invert(np.isnan(df['diff_REC']))]
df = df[np.invert(np.isinf(df['diff_REC']))]
df = df[np.invert(np.isnan(df['diff_VBPI']))]
df = df[np.invert(np.isinf(df['diff_VBPI']))]


# Para eleminar outfitting:
df = df[df['diff_REC']>(df['diff_REC'].mean()-3*df['diff_REC'].std())]
df = df[df['diff_REC']<(df['diff_REC'].mean()+3*df['diff_REC'].std())]
df = df[df['diff_COI']>(df['diff_COI'].mean()-3*df['diff_COI'].std())]
df = df[df['diff_COI']<(df['diff_COI'].mean()+3*df['diff_COI'].std())]
df = df[df['diff_VBPI']<(df['diff_VBPI'].mean()+3*df['diff_VBPI'].std())]
df = df[df['diff_VBPI']>(df['diff_VBPI'].mean()-3*df['diff_VBPI'].std())]

# Matriz de correlação
correlation = df.corr()
# Se quiser observar a matriz de correlação:
# seaborn.heatmap(correlation)
# plt.show()

# Análise de correlação
corrREC= correlation['diff_REC'][:-8]
corrCOI= correlation['diff_COI'][:-8]
corrVBPI= correlation['diff_VBPI'][:-8]

corrREC=corrREC.abs()
corrCOI=corrCOI.abs()
corrVBPI=corrVBPI.abs()

corrREC.sort_values(inplace=True)
corrCOI.sort_values(inplace=True)
corrVBPI.sort_values(inplace=True)


# Escolha das features para os modelos:

featREC = list(corrREC.index[-10:]) 
featCOI = list(corrCOI.index[-10:]) 
featVBPI = list(corrVBPI.index[-10:]) 

# Construção e teste dos modelos:
Xr = df[featREC]
Xc = df[featCOI]
Xv = df[featVBPI]
yr = df['diff_REC']
yc = df['diff_COI']
yv = df['diff_VBPI']

	# O modelo de regreção linear precisa da adição de um termo constante
const = pd.DataFrame({'const':np.ones(len(Xr))},index = Xr.index)

resultsCOI = sm.OLS(yc, Xc.join(const)).fit()
resultsREC = sm.OLS(yr, Xr.join(const)).fit()
resultsVBPI = sm.OLS(yv, Xv.join(const)).fit()
# Para acessar os resultados basta:
resultsCOI.summary()
resultsREC.summary()
resultsVBPI.summary()

NN = MLPRegressor(hidden_layer_sizes=(10,10,10),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# Resultados de R² para as Redes Neurais:
NN.fit(Xr,yr)
NN.score(Xr,yr)
NN.fit(Xc,yc)
NN.score(Xc,yc)
NN.fit(Xv,yv)
NN.score(Xv,yv)

# Considerar novas variáveis:
seems_relev = ['V0058_p', 'V0081_p', 'V0082_p', 'V0084_p', 'V0086_p', 'V0087_p', 'V0089_p', 'Lim_Simples', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33']
	
new_features=[]
for feature in seems_relev:
	if np.isinf(df[feature]).mean()>0:
		df[feature+'Inf'] = np.isinf(df[feature])*1
		new_features.append(feature+'Inf')
	if np.isnan(df[feature]).mean()>0:
		df[feature+'NaN'] = np.isnan(df[feature])*1
		new_features.append(feature+'NaN')
	df[feature] = df[feature].replace(np.inf, np.nan)
	df[feature] = df[feature].fillna(0)

want_see = seems_relev+new_features

print new_features


# Novos resultados para modelos com adição de novas variáveis :
Xc = df[featCOI+want_see[:8]]
Xr = df[featREC+want_see[:8]]
Xv = df[featVBPI+want_see[:8]]

resultsCOI = sm.OLS(yc, Xc.join(const)).fit()
resultsREC = sm.OLS(yr, Xr.join(const)).fit()
resultsVBPI = sm.OLS(yv, Xv.join(const)).fit()


print ''
print 'R² para Linear Regression com adição de novas variáveis ao modelo:'
print resultsCOI.rsquared
print resultsREC.rsquared
print resultsVBPI.rsquared

print '' 
print 'R² para Rede Neurais com adição de novas variáveis ao modelo:'
NN.fit(Xc,yc)
print NN.score(Xc,yc)
NN.fit(Xr,yr)
print NN.score(Xr,yr)
NN.fit(Xv,yv)
print NN.score(Xv,yv)

# Lucas' Model:
class LucasModel:
	def __init__(self,x,y,delt,feat_names=None,targ_names=None):
		self.X = x
		self.Y = y
		if feat_names:
			self.fN = feat_names
		else:
			self.fN = self.X.columns
		if targ_names:
			self.tN = targ_names
		else:
			self.tN = self.Y.columns
		self.d =(y.max()-y.min())*delt
	def fit(self):
		X=self.X
		Y=self.Y	
		tN=self.tN
		fN=self.fN
		Table = {}
		MU = Y.mean()
		STD = Y.std()
		PROB = 2*(norm.cdf(MU+self.d,MU,STD)-0.5)
		PROB = {tN[i]:PROB[i] for i in range(len(tN))}
		for t in tN:
			Table[t] = {}
			for f in fN:
				Table[t][f] = {}
				for i in (0,1):
					data = Y[t][X[f]==i]
					if len(data)>10:
						result = normaltest(data)
						if result[1] <0.01:
							mu = data.mean()
							std = data.std ()
							prob = 2*(norm.cdf(mu+self.d[t],mu,std)-0.5)
							Table[t][f][i] = {'E':mu,'p':prob}
						else:
							Table[t][f][i] = {'E':MU[t],'p':PROB[t]}
					else:
						
						Table[t][f][i] = {'E':MU[t],'p':PROB[t]}
		self.Table = Table
		return Table
	def predict(self,df):
		Table = self.Table
		tN = self.tN
		fN = self.fN
		estimation = {t:[] for t in tN}
		n = len(df)
		for i in range(n):
			print i,' of ',n  
			for t in tN:
				esperanca = 0
				normalize = 0
				for f in fN:	
					mu = Table[t][f][int(df[i:i+1][f])]['E']
					prob = Table[t][f][int(df[i:i+1][f])]['p']
					esperanca += mu*prob
					normalize += prob
				valor_esperado = esperanca/normalize
				estimation[t].append(valor_esperado)
		for t in tN:
			df[t+'_estm'] = estimation[t]
		return df				

Y = df[['diff_COI','diff_REC','diff_VBPI']]
lm = LucasModel(df[want_see[8:]],Y,0.01)
lm.fit()
df = lm.predict(df)


# Resulta com estimador:
Xc = df[featCOI+want_see[:8]+list(df.columns[-3:])]
Xr = df[featREC+want_see[:8]+list(df.columns[-3:])]
Xv = df[featVBPI+want_see+list(df.columns[-3:])]

resultsCOI = sm.OLS(yc, Xc.join(const)).fit()
resultsREC = sm.OLS(yr, Xr.join(const)).fit()
resultsVBPI = sm.OLS(yv, Xv.join(const)).fit()


print ''
print 'R² para Linear Regression com adição do estimador:'
print resultsCOI.rsquared
print resultsREC.rsquared
print resultsVBPI.rsquared

print '' 
print 'R² para Rede Neurais com adição do estimador:'
NN.fit(Xc,yc)
print NN.score(Xc,yc)
NN.fit(Xr,yr)
print NN.score(Xr,yr)
NN.fit(Xv,yv)
print NN.score(Xv,yv)


# Teste de Acurácia:

rec_feat= ['Lim_Simples','Desp_Op','COI_N','REC_N','VTI_N','VBPI_N','diff_REC_estm']
vbpi_feat = ['V0033_p','Gas_Pess','Desp_Op','VTI_N','COI_N','REC_N','VBPI_N','Lim_Simples','diff_VBPI_estm']
coi_feat = ['Lim_Simples','COI_N' ,'Desp_Op','VBPI_N','VTI_N','diff_COI_estm']


# Data Split
NN = MLPRegressor(hidden_layer_sizes=(10,10,10),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


Xc = df[coi_feat][:-218]
Xr = df[rec_feat][:-218]
Xv = df[vbpi_feat][:-218]
Yr = df['diff_REC'][:-218]
Yc = df['diff_COI'][:-218]
Yv = df['diff_VBPI'][:-218]
yr = df['diff_REC'][-218:]
yc = df['diff_COI'][-218:]
yv = df['diff_VBPI'][-218:]
xc = df[coi_feat][-218:]
xr = df[rec_feat][-218:]
xv = df[vbpi_feat][-218:]
NN.fit(Xc,list(Yc))
print NN.score(xc,yc)
NN.fit(Xr,list(Yr))
print NN.score(xr,yr)
NN.fit(Xv,list(Yv))
print NN.score(xv,yv)

#CrosValidation k-fold (k=10)

scores = cross_val_score(NN, df[vbpi_feat], df['diff_VBPI'], cv=10)
print scores
print scores.mean()
scores = cross_val_score(NN, df[rec_feat], df['diff_REC'], cv=10)
print scores
print scores.mean()
scores = cross_val_score(NN, df[coi_feat], df['diff_COI'], cv=10)
print scores
print scores.mean()

# Para gerar os plots dos fetures alterados e novos features:
#f, axarray = plt.subplots(6, 6, sharex='col', sharey='row')
#targets = ['diff_COI','diff_REC','diff_VBPI']
#for i in range(6):
#	for j in range (6):
#		axarray[i][j].scatter(df[targets[j%3]],df[want_see[28+i*2+j/3]], marker = '.')
#		axarray[i][j].set_title(want_see[28+i*2+j/3]+' '+targets[j%3])

