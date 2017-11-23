# Medical Data Regression - Lucas Armand
Este trabalho é um relatório que discute e analisa diversas técnicas de regressão e classificação realizadas sobre uma base de dados médicos reais a respeito da capacidade vascular dos pacientes, ele foi construído para avaliação dos professores Raimundo, Rosa e Daniel para compor o grau na disciplina Aprendizado de Máquina - CPS863 - 2017/3.

![Questions](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/questions.png)


## Visualização dos Dados:

Inicialmente devemos construir um conhecimento mínimo dos dados que desejamos analisar, isso é feita através de uma análise exploratória dos dados. A seguir utilizaremos alguma tecnicas de visualização para construir um entendimento sobre os dados

A base de dados é composta por 1172 pontos cada um com quatro características: Idade, Carga Final, Peso e VO2 (os dois primeiros são variáveis inteiras e as demais são definidas no espaço do números reais). A seguir podemos ver os primeiros cinco pontos da base de dados:

```
    IDADE (anos)  Peso (kg)  Carga Final  VO2 medido máximo (mL/kg/min)
0            49       79.1          250                      49.051833
1            30       52.4          177                      41.603053
2            56       65.8          140                      32.674772
3            29       78.0          400                      59.102564
4            49       69.2          242                      48.410405

```

Algumas informações são importantes para entender os dados, a seguir esta apresentado um pequeno sumário sobre os dados da variáveis:

```
       IDADE (anos)    Peso (kg)  Carga Final  VO2 medido máximo (mL/kg/min)
count   1172.000000  1172.000000  1172.000000                    1172.000000
mean      53.290956    85.925776   172.271502                      29.394728
std       14.746297    14.799113    70.093124                      10.497250
min       18.000000    45.300000    30.000000                       5.846847
0%        18.000000    45.300000    30.000000                       5.846847
50%       54.000000    83.700000   170.000000                      28.326660
max       91.000000   178.900000   432.000000                      73.333333

```

Na tabela a cima podemos ver o máximo e o mínimo das cadaracterísticas armazenadas na base, o que nos permite conhecer um pouco sobre os individuo. Eles são homens entre 18 e 91 anos que pesam em média 86kg com desvio padrão de 15kg. A "Carga Final" varia entre [70,430] e "V02" entre [6,70]. Outro informação relevante sobre a base é que não existe valores indeterminados ("nan","NA", "inf",etc...) em nenhum ponto, isso facilitará análises futuras.

Para entender melhor como esses dados estão distribuídos podemos construir histogramas da base de dados:

![Histograma das Variáveis da Base de Dados](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/hist.png)

No histograma é possível perceber que em todas as características tem distribuições "bem comportadas" aparentando ter uma distribuição próxima a uma normal ou uma distribuição gamma. 

Outra análise que pode apresentar respostas interessantes é procurar correlações entre os dados,por isso será feito uma apresentação das correlações lineares entre as variáveis (por tabela de dados e 'heatmap') seguida de uma apresentação de um scaterplot da base de dados:

```
                               IDADE (anos)  Peso (kg)  Carga Final     V02 médio máximo(mL/kg/min)
IDADE (anos)                       1.000000  -0.146315    -0.692058             -0.630072
Peso (kg)                         -0.146315   1.000000     0.186422             -0.174401 
Carga Final                       -0.692058   0.186422     1.000000              0.878326
VO2 medido máximo (mL/kg/min)     -0.630072  -0.174401     0.878326              1.000000 
```
![HeatMap das correlações](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/correlation_heatmap.png)

![ScatterPlot todas as variáveis](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/scatter_all.png)

Os resultados apresentados nessa última parte são bem interessantes as variáveis Idade, Carga e VO2 apresentaram um alto nível de correlação entre si enquanto Peso parece ser mais independente (num primeiro momento).

Assim, uma vez que uma exploração inicial dos dados foi realizada discussões a respeito de modelos podem ser executadas com uma maior propriedade.

# Data Split:

Inicialmente faremos uma divisão na base dados em dados de teste e dados de treinamento, de maneira que 172 dados sejam separados para dados de teste, assim foi gerado um vector com (X) que possui 1172 valores binários aonde mil valores são do tipo "1" e 172 são do tipo "0". Os resultados nesse vetor foram embaralhados aleatoriamente de maneira que, em nossa base, todos os dados para qual o valor de "x" for igual a "1" então o ponto de ordenação correspondente na base dado será definido com ponto de treinamento, de maneira análoga, os pontos que estiverem associados a um "0" no vetor X serão definidos como pontos de teste. Para referência, segue o vetor X na seção de Apêndice desse mesmo trabalho.

Feito isso temos dois novos conjuntos:

```
>>> df.describe()
       IDADE (anos)    Peso (kg)  Carga Final  VO2 medido máximo (mL/kg/min)
count   1172.000000  1172.000000  1172.000000                    1172.000000
mean      53.290956    85.925776   172.271502                      29.394728
std       14.746297    14.799113    70.093124                      10.497250
min       18.000000    45.300000    30.000000                       5.846847
25%       42.000000    76.100000   120.000000                      21.797423
50%       54.000000    83.700000   170.000000                      28.326660
75%       64.000000    94.450000   220.000000                      35.853793
max       91.000000   178.900000   432.000000                      73.333333

>>> df_trnn.describe()
       IDADE (anos)    Peso (kg)  Carga Final  VO2 medido máximo (mL/kg/min)
count   1000.000000  1000.000000  1000.000000                    1000.000000
mean      53.499000    85.812210   170.897700                      29.187956
std       14.804431    14.737553    69.747198                      10.515372
min       18.000000    45.300000    30.000000                       5.846847
25%       42.000000    75.900000   120.000000                      21.550613
50%       54.000000    83.750000   167.500000                      27.932131
75%       64.250000    94.125000   220.000000                      35.742980
max       91.000000   178.900000   432.000000                      73.333333

>>> df_test.describe()
       IDADE (anos)   Peso (kg)  Carga Final  VO2 medido máximo (mL/kg/min)
count    172.000000  172.000000   172.000000                     172.000000
mean      52.081395   86.586047   180.258721                      30.596890
std       14.386247   15.179042    71.757785                      10.339651
min       23.000000   52.400000    40.000000                      10.285006
25%       41.000000   77.150000   130.000000                      22.752700
50%       53.500000   83.550000   177.000000                      29.349168
75%       62.250000   95.575000   220.000000                      37.547858
max       86.000000  132.500000   400.000000                      61.498029
```
As tabelas apresentam as característica dos novos conjuntos de pontos. Testes mais avançados poderiam (e deveriam ser feito para um problema real), afim de garantir que os conjuntos sejam representativos da população (evitando assim resultados com bias), mas, na visão do autor, os resultados mostrados, se não suficientes para garantir, corraboram para hipótese de que essa divisão de dados seja coerente.

# I) Regressão de VO2:

1) O primeiro modelo proposto é uma regressão polinomial de "VO2" por "Carga Final". A seguir faremos testes com polinômios entre grau 1 e 7. Polinômios com grau superior poderiam ter sido utilizados, mas sete variações de modelo polinomial são suficientes para embasar uma análise exploratória desse tipo de modelo e utilizar mais variações do modelo acarretaria numa poluição das imagens. Assim a seguir temos um plot dos resultados obtidos:

![Plots para a regressão polinomial](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/poly_plot.png)

Resultados das regressões estão no apêndice, assim como os sumários das regressões. Os coeficientes dos modelos estão plotados na tabela a seguir:

![Coeficiente dos termos na primeira regressão linear](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/Coef_linear_1.png)

A seguir segue uma breve discussão sobre a relevância do NLL(w) num estudo do caso do modelo de regressão linear:

![Imagens scaneadas do caderno](?)

Uma vez entendido o que é o NLL é possível se compreender como os modelos (que os resultados foram plotados no inicio dessa seção) foram obtidos. Aqui talvez fosse interessante apresentar o NLL para cada modelo, mas foi feito a opção por usar o R² para compara os resultados do múltiplos modelos. Isso porque, na visão do autor, ele oferece um resultado de mais fácil compreensão sobre a capacidade explicativa dos modelos, além de ser "normalizado" (a ordem do resultado não é influencia pela número de pontos nem pela ordem de grandeza de "y". R² é definido como:

![R-Squared definição](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/raquared.png)

Aqui fica claro como o R² se relaciona com o NLL uma vez que ambos são baseados no RSS (Residual Sum of Squares). Um plot da variação do R² (dos dados de teste e treinamento) para os modelos testados:

![Comparação resultados para dados de treinamento e de teste](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/testXtrain_result.png)

```
y1=[0.7750754467,0.7754524946,0.7754539306,0.7757899819,0.7759189671,0.775985176,0.7760899583]
y2=[0.7454585045,0.7455548709,0.7454815363,0.7451629196,0.7436750232,0.7439609531,0.7434698284]
```

Nesse gráfico é possível perceber que a medida que o número de features, para suportar o modelo, aumenta o R² nos dados de treinamento aumenta, porém o contrário é percebido para os dados de teste, nestes é possível ver uma pequena melhora seguida de uma queda no resultado.

Se considerarmos simplesmente os resultados, o modelo que apresenta o maior poder explicativo sobre os resultados sobre os dados de treinamento é o modelo com termos até a segunda ordem, porém a variação dos resultados foi tão pequena que, na visão do autor, o melhor modelo para predizer os dados é o de uma função de primeiro grau. Outros tipo de testes de acurácia poderiam ser feitos para aumentar a confiança nos resultados (em vez de fazer um simples split dos dados poderia se usar um K-fold), mas como os melhores foi de uma diferença muito pequena, foi dado preferencia ao mais simples.

Assim, usando só a carga como input, o modelo escolhido é:
 
![R-Squared definição](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/model1.png)

2) A seguir repetiremos a análise, mas utilizando as variáveis de "Peso" e "Carga" para regredir "VO2":

![Comparação resultados para dados de treinamento e de teste](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/r_square_2.png)

Como podemos ver na imagem a cima, existe uma região aonde os modelos obtem o melhor resultado. Essa região se inicia nas funções com termos de segundo grau, por isso o modelo escolhido para regressão de VO2 a partir de Carga e Peso:

![modelo 2](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/model2.png)

3) Agora uma discussão análoga pode ser executada para uma regressão de "Idade","Peso" e "Carga":

![Comparação resultados para dados de treinamento e de teste](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/r_square_3.png)

Nos resultados apresentados a cima é possível perceber que a adição da variável "Idade" não aumentou o nível de acurácia dos modelos, de maneira que o modelo adotado para regressão com as três variáveis será o mesmo na etapa anterior, ou seja:


![modelo 2](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/model2.png)


4) Agora os resultados obtidos deve ser comparado com um modelo baseado na teoria médica:

Se compararmos o modelo escolhido como melhor dentre todas as regressões com o proposto pela "American College of Sports
Medicine" obteremos o seguinte resultado:

```
>>> r2_score(yr34,y_ACSM)
0.90139352984810062
>>> r2_score(yr234,y_myModel)
0.89900747095256661

```
A pesar dos resultados serem praticamente equivalentes o modelo que apresenta maior poder explicativo deve ser sempre preferido, assim o modelo proposto pelos especialista tem maior valor.

# II)  Gaussiana multivariada:

## 1) Modelo de Gaussiana multivariada bidimensional
O próximo modelo é uma gaussiana multivariada. Na teoria de probabilidade e nas estatísticas, a distribuição normal multivariada (ou distribuição gaussiana multivariada) é uma generalização da distribuição normal uni variada para maiores dimensões. A Hipótese básica que as features são normalmente distribuidas (numa análise univariável) e que são linearmente correlacionadas. 

Se realizarmos uma regressão utilizando as variáveis "Carga Final" e "VO2 medido" como base de dados,  usando os dados de treinamento, chegaremos aos resultados:

```
>>> mean
Carga Final                      170.897700
VO2 medido máximo (mL/kg/min)     29.187956
dtype: float64

>>> covarr
                               Carga Final  VO2 medido máximo (mL/kg/min)
Carga Final                    4864.671596                     645.688994
VO2 medido máximo (mL/kg/min)   645.688994                     110.573043

```
Aonde os parâmetros do modelo são as médias das features e a matriz de covariância (análogo ao caso univariável).
De maneira que se plotarmos o modelo sobre a base de dados teremos o seguinte resultado:

![Modelo de Gaussiana Multivariada 2 dimessões](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/model_gauss_plus_scatter.png)

Um exemplo bidimensional é interessante justamente por que é possível realizar representações como essas apresentadas a cima. Nessas imagens é possível perceber que aparentemente o modelo é razoavelmente adequado aos dodos. A seguir implementações com uma dimensionalidades maiores serão feitas e dai será discutido a acurácia e outros aspectos relevantes desse tipo de modelo.

## 2) Modelo de Gaussiana multivariada 3-dimensional

A seguir iremos regredir o mesmo tipo de modelo usando como base "Peso (kg)", "Carga Final" e "VO2 medido máximo (mL/kg/min)":

```
mean:
Peso (kg)                         85.812210
Carga Final                      170.897700
VO2 medido máximo (mL/kg/min)     29.187956

covariance:
                                Peso (kg)  Carga Final  VO2 medido máximo (mL/kg/min)
Peso (kg)                      217.195456   183.404173         -27.332390
Carga Final                    183.404173  4864.671596         645.688994
VO2 medido máximo (mL/kg/min)  -27.332390   645.688994         110.573043 

```
Para entender um pouco mais as propriedades desse tipo de modelo vamos olhar para os resultados em alguns pontos: 

![Pontos escolhidos](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/pontos_gauss_quetion2.png)

Arbitrariamente foram selecionados os pontos:

Ponto | Peso | Carga
----  | ---- | -----
 1    | 60   | 100
 2    | 80   | 200
 3    | 100  | 300

Para esses pontos temos as seguintes distribuições de VO2:

Ponto 1:

![Plot 1](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/plot_V02_60_100.png)

Ponto 2:

![Plot 2](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/plot_V02_80_200.png)

Ponto 3:

![Plot 3](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/plot_V02_100_300.png)

Essas funções de probabilidade são obtidas através de probabilidade condicional, ou seja, sabendo que a função de probabilidade da gaussiana multivariada é:

![Formula da destribuição gaussiana](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/gauss_form.png)

Considerando a probabilidade condicional:

![Probabilidade condicional](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/condicional_form.png)

Dessa maneira obtemos as funções de probabilidade para os pontos citados a cima. Porém, muitas das vezes não se deseja obter a distribuição em si, mas se realizar alguma sorte de predição sobre a variável. Nesse sentido esse método é diferente do analisado anteriormente porque além de ser possível escolher o valor mais provável (no caso a média) para ser o "resultado" da regressão, mas é possível estabelecer um nível de confiança se a esse resultado for associado um intervalo e a probabilidade do resultado estar dentro desse intervalo. A seguir uma tabela que apresenta o resultado obtidos a partir dos dois modelos explorados até esse ponto.


Linear| Gauss média| Gauss interv.
----  | ---- | -----
26.91 | 25.46  | 0.7952
35.24 | 34.74   | 0.7952
42.70 | 44.03  | 0.7952
 


# Apêndice:


Valor do vector "X" que contem a ordenação dos vetores de teste e de treinamento:


X = [1 , 0 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]


Summario dos Modelos de Regressão linear:



```
                                  OLS Regression Results                                  
==========================================================================================
Dep. Variable:     VO2 medido máximo (mL/kg/min)   R-squared:                       0.775
Model:                                        OLS   Adj. R-squared:                  0.775
Method:                             Least Squares   F-statistic:                     3439.
Date:                            Sex, 10 Nov 2017   Prob (F-statistic):               0.00
Time:                                    11:52:58   Log-Likelihood:                -3025.3
No. Observations:                            1000   AIC:                             6055.
Df Residuals:                                 998   BIC:                             6064.
Df Model:                                       1                                         
Covariance Type:                        nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------
Carga Final**1     0.1327      0.002     58.643      0.000         0.128     0.137
const              6.5047      0.418     15.571      0.000         5.685     7.324
==============================================================================
Omnibus:                       49.019   Durbin-Watson:                   2.062
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               85.515
Skew:                           0.366   Prob(JB):                     2.70e-19
Kurtosis:                       4.232   Cond. No.                         489.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

```


```
                                  OLS Regression Results                                  
==========================================================================================
Dep. Variable:     VO2 medido máximo (mL/kg/min)   R-squared:                       0.775
Model:                                        OLS   Adj. R-squared:                  0.775
Method:                             Least Squares   F-statistic:                     1722.
Date:                            Sex, 10 Nov 2017   Prob (F-statistic):          4.94e-324
Time:                                    11:52:58   Log-Likelihood:                -3024.4
No. Observations:                            1000   AIC:                             6055.
Df Residuals:                                 997   BIC:                             6070.
Df Model:                                       2                                         
Covariance Type:                        nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------
Carga Final**1     0.1445      0.009     15.465      0.000         0.126     0.163
Carga Final**2 -3.175e-05   2.45e-05     -1.294      0.196     -7.99e-05  1.64e-05
const              5.5824      0.826      6.757      0.000         3.961     7.204
==============================================================================
Omnibus:                       49.936   Durbin-Watson:                   2.064
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               87.613
Skew:                           0.370   Prob(JB):                     9.44e-20
Kurtosis:                       4.247   Cond. No.                     2.26e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.26e+05. This might indicate that there are
strong multicollinearity or other numerical problems.

```


```
                                  OLS Regression Results                                  
==========================================================================================
Dep. Variable:     VO2 medido máximo (mL/kg/min)   R-squared:                       0.775
Model:                                        OLS   Adj. R-squared:                  0.775
Method:                             Least Squares   F-statistic:                     1147.
Date:                            Sex, 10 Nov 2017   Prob (F-statistic):          1.98e-322
Time:                                    11:52:58   Log-Likelihood:                -3024.4
No. Observations:                            1000   AIC:                             6057.
Df Residuals:                                 996   BIC:                             6077.
Df Model:                                       3                                         
Covariance Type:                        nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------
Carga Final**1     0.1465      0.027      5.440      0.000         0.094     0.199
Carga Final**2 -4.315e-05      0.000     -0.298      0.766        -0.000     0.000
Carga Final**3  1.894e-08   2.37e-07      0.080      0.936     -4.47e-07  4.85e-07
const              5.4820      1.505      3.642      0.000         2.528     8.436
==============================================================================
Omnibus:                       49.973   Durbin-Watson:                   2.065
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               87.670
Skew:                           0.370   Prob(JB):                     9.18e-20
Kurtosis:                       4.247   Cond. No.                     1.12e+08
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.12e+08. This might indicate that there are
strong multicollinearity or other numerical problems.

```


```
                                  OLS Regression Results                                  
==========================================================================================
Dep. Variable:     VO2 medido máximo (mL/kg/min)   R-squared:                       0.776
Model:                                        OLS   Adj. R-squared:                  0.775
Method:                             Least Squares   F-statistic:                     860.7
Date:                            Sex, 10 Nov 2017   Prob (F-statistic):          3.45e-321
Time:                                    11:52:58   Log-Likelihood:                -3023.7
No. Observations:                            1000   AIC:                             6057.
Df Residuals:                                 995   BIC:                             6082.
Df Model:                                       4                                         
Covariance Type:                        nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------
Carga Final**1     0.2202      0.066      3.331      0.001         0.090     0.350
Carga Final**2    -0.0007      0.001     -1.257      0.209        -0.002     0.000
Carga Final**3  2.333e-06   1.91e-06      1.222      0.222     -1.41e-06  6.08e-06
Carga Final**4 -2.752e-09   2.25e-09     -1.221      0.222     -7.17e-09  1.67e-09
const              2.8497      2.629      1.084      0.279        -2.309     8.008
==============================================================================
Omnibus:                       49.044   Durbin-Watson:                   2.066
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               86.644
Skew:                           0.362   Prob(JB):                     1.53e-19
Kurtosis:                       4.247   Cond. No.                     5.99e+10
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.99e+10. This might indicate that there are
strong multicollinearity or other numerical problems.

```


```
                                  OLS Regression Results                                  
==========================================================================================
Dep. Variable:     VO2 medido máximo (mL/kg/min)   R-squared:                       0.776
Model:                                        OLS   Adj. R-squared:                  0.775
Method:                             Least Squares   F-statistic:                     688.4
Date:                            Sex, 10 Nov 2017   Prob (F-statistic):          8.11e-320
Time:                                    11:52:58   Log-Likelihood:                -3023.4
No. Observations:                            1000   AIC:                             6059.
Df Residuals:                                 994   BIC:                             6088.
Df Model:                                       5                                         
Covariance Type:                        nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------
Carga Final**1     0.1206      0.147      0.818      0.414        -0.169     0.410
Carga Final**2     0.0005      0.002      0.311      0.756        -0.003     0.004
Carga Final**3 -4.527e-06   9.27e-06     -0.488      0.625     -2.27e-05  1.37e-05
Carga Final**4  1.443e-08   2.28e-08      0.632      0.527     -3.04e-08  5.92e-08
Carga Final**5 -1.585e-11   2.09e-11     -0.756      0.450      -5.7e-11  2.53e-11
const              5.6023      4.490      1.248      0.212        -3.208    14.412
==============================================================================
Omnibus:                       48.771   Durbin-Watson:                   2.066
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               86.819
Skew:                           0.357   Prob(JB):                     1.40e-19
Kurtosis:                       4.254   Cond. No.                     3.41e+13
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.41e+13. This might indicate that there are
strong multicollinearity or other numerical problems.

```


```
                                  OLS Regression Results                                  
==========================================================================================
Dep. Variable:     VO2 medido máximo (mL/kg/min)   R-squared:                       0.776
Model:                                        OLS   Adj. R-squared:                  0.775
Method:                             Least Squares   F-statistic:                     688.6
Date:                            Sex, 10 Nov 2017   Prob (F-statistic):          7.34e-320
Time:                                    11:52:58   Log-Likelihood:                -3023.3
No. Observations:                            1000   AIC:                             6059.
Df Residuals:                                 994   BIC:                             6088.
Df Model:                                       5                                         
Covariance Type:                        nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------
Carga Final**1     0.3615      0.055      6.630      0.000         0.254     0.468
Carga Final**2    -0.0033      0.001     -2.241      0.025        -0.006    -0.000
Carga Final**3  2.414e-05   1.44e-05      1.675      0.094     -4.15e-06  5.24e-05
Carga Final**4 -9.668e-08   6.67e-08     -1.449      0.148     -2.28e-07  3.42e-08
Carga Final**5  1.968e-10   1.45e-10      1.355      0.176     -8.82e-11  4.82e-10
Carga Final**6 -1.586e-13    1.2e-13     -1.323      0.186     -3.94e-13  7.67e-14
const              0.0146      0.002      6.628      0.000         0.010     0.019
==============================================================================
Omnibus:                       49.517   Durbin-Watson:                   2.066
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               88.047
Skew:                           0.363   Prob(JB):                     7.60e-20
Kurtosis:                       4.260   Cond. No.                     2.03e+16
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.03e+16. This might indicate that there are
strong multicollinearity or other numerical problems.

```
```
                                  OLS Regression Results                                  
==========================================================================================
Dep. Variable:     VO2 medido máximo (mL/kg/min)   R-squared:                       0.773
Model:                                        OLS   Adj. R-squared:                  0.772
Method:                             Least Squares   F-statistic:                     678.3
Date:                            Sex, 10 Nov 2017   Prob (F-statistic):          2.35e-317
Time:                                    11:52:58   Log-Likelihood:                -3029.1
No. Observations:                            1000   AIC:                             6070.
Df Residuals:                                 994   BIC:                             6100.
Df Model:                                       5                                         
Covariance Type:                        nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------
Carga Final**1     0.0003   1.89e-05     16.502      0.000         0.000     0.000
Carga Final**2     0.0091      0.001     16.503      0.000         0.008     0.010
Carga Final**3    -0.0001   1.25e-05    -10.839      0.000        -0.000    -0.000
Carga Final**4  9.186e-07   1.08e-07      8.508      0.000      7.07e-07  1.13e-06
Carga Final**5 -3.199e-09   4.45e-10     -7.197      0.000     -4.07e-09 -2.33e-09
Carga Final**6  5.569e-12   8.76e-13      6.355      0.000      3.85e-12  7.29e-12
Carga Final**7 -3.835e-15   6.64e-16     -5.772      0.000     -5.14e-15 -2.53e-15
const           6.733e-06   4.08e-07     16.501      0.000      5.93e-06  7.53e-06
==============================================================================
Omnibus:                       47.552   Durbin-Watson:                   2.045
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               79.745
Skew:                           0.368   Prob(JB):                     4.83e-18
Kurtosis:                       4.171   Cond. No.                     1.26e+19
==============================================================================

```

Resultado da regreção para dados de treinamento e dados de teste:
Grau 1:
R² Trainamento:
0.775075446748
R² Teste:
0.745458504467

Grau 2:
R² Trainamento:
0.775452494623
R² Teste:
0.745554870891

Grau 3:
R² Trainamento:
0.775453930569
R² Teste:
0.745481536333

Grau 4:
R² Trainamento:
0.775789981862
R² Teste:
0.745162919559

Grau 5:
R² Trainamento:
0.77591896708
R² Teste:
0.743675023225

Grau 6:
R² Trainamento:
0.77598517595
R² Teste:
0.743960953077

Grau 7:
R² Trainamento:
0.776089958278
R² Teste:
0.743469828356
