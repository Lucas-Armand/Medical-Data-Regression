# Medical-Data-Regression
Análise de diversars técincas de regressão e classificassão realizadas sobre uma base de dados médicos reais a respeito da capacidade vascular dos pacientes.

"Nesta lista, iremos utilizar dados reais fornecidos gentilmente pelo Professor Claudio Gil Soares de Araujo (at ́e recentemente professor do Instituto do Cora ̧c ̃ao Edson Saad da UFRJ) da CLINIMEX,
atrav ́es da aluna de doutorado da UFRJ Christina G. de Souza e Silva.  Os dados foram obtidos a partir  de  uma  extensa  base  de  dados  do  Prof.   Claudio  Gil,  coletada  durante  muitos  anos  e  usada
em suas pesquisas.  Os dados mostram uma medida da condi ̧c ̃ao aer ́obica do paciente (o VO2 max) (por quilo de peso do indiv ́ıduo) e ainda as vari ́aveis idade, peso e a carga m ́axima atingida durante
um teste de exerc ́ıcio  m ́aximo ao qual o paciente foi submetido.  (Os dados s ̃ao todos de pacientes masculinos.)   De  forma  bem  simples,  o  VOD2
max  ́e  a  taxa  m ́axima  de  consumo  de  oxigˆenio  medida durante  um  teste  de  exerc ́ıcio  m ́aximo,  e  reflete  a  capacidade  aer ́obica  do  paciente,  expressa  em
volume de oxigˆenio por massa corporal por minuto (ml/(Kg.min). E uma importante m ́etrica usada
na avalia ̧c ̃ao cardiovascular de indiv ́ıduos [1].

Nesta  lista  os  modelos  devem  prever  VO2 m ́aximo  de  pacientes  com  uma  dada  idade,  peso  e carga  m ́axima  atingida  durante  o  teste  de
exerc ́ıcio  m ́aximo do  paciente.   Em  alguns  dos  modelos vocˆe dever ́a encontrar uma fun ̧c ̃ao adequada do VO2 m ́aximo em fun ̧c ̃ao das outras vari ́aveis (ou de
um subconjunto delas).  Outros modelos s ̃ao mais evidentes para se encontrar a probabilidade de se encontrar o VO 2 m ́aximo dentro de uma faixa de valores,  a partir dos dados de entrada ou de um
subconjunto deles.  Em outra quest ̃ao ser ́a solicitado que seja estimada a idade do paciente dado um subconjunto de vari ́aveis."

## Vizualização dos Dados:

Inicialmente devemos construir um conhecimento mínimo dos dados que desejamos analisar, isso é feita atravez de uma análise exploratória dos dados. A seguir utilizaremos alguma tecnicas de vizualização para construir um entendimento sobre os dados

A base de dados é composta por 1172 pontos cada um com quatro caracteristicas: Idade, Carga Final, Peso e VO2 (os dois primeros são variáveis inteiras e as demais são definidas no espaço do números reais). A seguir podemos ver os primeiros cinco pontos da base de dados:

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

Para entender melhor como esses dados estão destribuidos podemos construir histogramas da base de dados:

![Histograma das Variáveis da Base de Dados](https://github.com/Lucas-Armand/Medical-Data-Regression/blob/master/hist.png)

No histograma é possível perceber que em todas as características tem destribuições "bem comportadas" aparentando ter uma distribuição próxima a uma normal ou uma destribuição gamma. 

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

Os resultados apresentados nessa última parte são bem interessantes as variáveis Idade, Carga e VO2 apresentaram um alto nível de correlação entre sí enquanto Peso parece ser mais independete (num primeiro momento).

Assim, uma vez que uma exploração inicial dos dados foi realizada discuções a respeito de modelos podem ser executadas com uma maior propriedade.

# Data Split:

Inicialmente faremos uma divisão na base dados em dados de teste e dados de trainamento, de maneira que 172 dados sejam separados para dados de teste, assim foi gerado um vector com (X) que possúi 1172 valores binários aonde mil valores são do tipo "1" e 172 são do tipo "0". Os resultados nesse vetor foram embaralhados aleatóriamente de maneira que, em nossa base, todos os dados para qual o valor de "x" for igual a "1" então o ponto de ordenação correspondente na base dado será defenido com ponto de treinamento, de maneira análoga, os pontos que estiverem associados a um "0" no vetor X serão definidos como pontos de teste. Para referência, segue o vetor X na seção de Apêndice desse mesmo trabalho.

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

# I) 

# Apêndice:



Valor do vector "X" que contem a ordenação dos vetores de teste e de treinamento:

```
X = [1 , 0 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]

```
