
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Avtive investment vs Pasive investment                                                        -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: if723286                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/if723286/Portafolio-Activo-Pasivo                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import os
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt


#%% directory containing the csv files
directory = 'files'

columns = ['Ticker']
df_global = pd.DataFrame(columns=columns)

# loop through all the files in the directory
for filename in os.listdir(directory):
    # check if the file is a csv file
    if filename.endswith(".csv"):
        # read the csv file and store it in a data frame
        df = pd.read_csv(os.path.join(directory, filename), skiprows=2, nrows=37 , usecols=["Ticker", 'Peso (%)', 'Precio'])
        # append the data frame to the list
        df_global = pd.concat([df_global, df])     

#Fixing data
df_global['Ticker'] = df_global['Ticker'].map(lambda x: x.replace('*',''))
df_global['Ticker'] = df_global['Ticker'].map(lambda x: x.replace('.',''))
df_global['Ticker'] = df_global['Ticker'].apply(lambda x: "{}{}".format(x, '.MX'))

#Get list of the stocks that appear all the times in the NASDAC
df_global_list = df_global['Ticker'].value_counts(sort=True,ascending=False).index[:34].tolist()

#Fix errors
df_global_list.remove('LIVEPOLC1.MX')
df_global_list.remove('MXN.MX')
df_global_list.append('LIVEPOLC-1.MX')
p_portfolio = df_global.head(33)
p_portfolio['Precio'] = p_portfolio['Precio'].astype(str)
p_portfolio['Precio'] = p_portfolio['Precio'].map(lambda x: x.replace(',',''))
p_portfolio['Precio'] = p_portfolio['Precio'].astype(float)
p_portfolio['Peso (%)'] = p_portfolio['Peso (%)'].apply(lambda x: x/100)

#%%PASIVE PORTFOLIO
p_portfolio = p_portfolio.sort_values('Ticker')
p_portfolio = p_portfolio.reset_index(drop=True)

#Lets get the value to invest in each and every stock
p_portfolio['Value'] = p_portfolio['Peso (%)'] * 1000000
january_2021 = p_portfolio.Value.sum()

#Lets call the prices of the stocks included in our portfolio
tickers = df_global_list
data = yf.download(tickers = tickers, start= '2021-01-01', end='2023-02-01', interval = "1mo")
data = data['Adj Close']
data = data.transpose()
data = data.reset_index(drop=True)

#Now include weight to every Ticker
data.insert(0, 'Peso (%)', p_portfolio['Peso (%)'])
data.insert(0, 'Ticker', p_portfolio['Ticker'])  
data = data.dropna()

#Start first line of final table
primero = pd.DataFrame({'capital': [january_2021]})

results=[]
for j in range (0,24):
    suma = 0
    for i in range (0, 31):
        X = 0
        X = (data.iloc[i,j+3]/data.iloc[i,j+2]-1)*data.iloc[i,1]
        if X>0:
            suma += X
    results.append(suma)

df = pd.DataFrame (results, columns = ['rendimiento'])
df.loc[-1] = [0]  # adding a row
df.index = df.index + 1  # shifting index
df.sort_index(inplace=True) 

#Set the date column
dates=[] 
# range of dates
date_range = pd.period_range(
    start='2021-01-01', periods=26, freq='M')
  
# timestamp range
timestamp_range = [x.to_timestamp() for x in date_range]
  
# iterating through timestamp range
for i in timestamp_range:
    dates.append(i)

df_pasiva = pd.DataFrame (dates, columns = ['timestamp'])

#add it to dataframe
df_pasiva.insert(1, 'capital', primero['capital'])
df_pasiva.insert(2, 'rendimiento', df['rendimiento'])

#fill the return for each date
df_pasiva.iloc[1,1] = df_pasiva.iloc[0,1] #(row,column)

for i in range (0,23):
    df_pasiva.iloc[i+2,1] = df_pasiva.iloc[i+1,1]*(1 + df_pasiva.iloc[i+1,2])

#Acumulated return
acumulado=[]
for i in range(0,23):
    x  = df_pasiva.iloc[i+1,1]/df_pasiva.iloc[0,1]-1
    acumulado.append(x)
acumulado = pd.DataFrame (acumulado, columns = ['rendimiento acumulado'])

#Clean en present
df_pasiva.insert(3, 'rendimiento acumulado', acumulado['rendimiento acumulado'])
df_pasiva = df_pasiva.dropna()

df_pasiva

#%% Active portfolio
#Import data and get returns
data = yf.download(tickers = tickers, start= '2021-01-01', end='2023-02-01', interval = "1mo")
data = data['Adj Close']
ret1 = data.pct_change().dropna()

#Lets plot it, why not?
ret1.plot(figsize=(9,7),grid=True);

# Rendimientos esperados, volatilidad, matriz de covarianza y matriz de correlacion
tabla1 = pd.DataFrame(data={'Media1':ret1.mean(),
                            'Volatilidad1':ret1.std()
                            }, index=ret1.columns)

cov1 = ret1.cov()
corr1 = ret1.corr()

#Tasa libre de riesgo
rf1 = 0.11 / 12

# Construcción de parámetros
S1 = np.diag(tabla1.iloc[:,1])
Sigma1 = S1.dot(corr1).dot(S1)

Eind1 = tabla1.iloc[:,0]

# Función objetivo
def varianza(w, Sigma1):
    return w.T.dot(Sigma1).dot(w)

# Dato inicial
n1 = len(Eind1)
w01 = np.ones(n1) / n1
# Cotas de las variables
bnds1 = ((0, 1),) * n1
# Restricciones
cons1 = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

# Portafolio de mínima varianza
minimavar1 = minimize(fun=varianza,
                   x0=w01,
                   args=(Sigma1,),
                   bounds=bnds1,
                   constraints=cons1
                  )

# Pesos, rendimiento, riesgo y Radio de Sharpe del portafolio de mínima varianza

w_minvar1 = minimavar1.x
E_minvar1 = Eind1.T.dot(w_minvar1)
s_minvar1 = varianza(w_minvar1, Sigma1)**0.5
RS_minvar1 = (E_minvar1 - rf1) / s_minvar1

#Save the weight of every and each stock
data = data.transpose()
data.insert(0, 'Peso (%)', w_minvar1)

#Create date column
df_activa = pd.DataFrame (dates, columns = ['timestamp'])

# Present it better
ret1=ret1.transpose()
ret1.insert(0, 'Peso (%)', w_minvar1)

#We only want those stocks that weight more than 0.1% of the portfolio, every other can be deleted
df = pd.DataFrame(ret1)
# Define the condition to filter the rows
condition = df['Peso (%)'] > 0.001

# Filter the DataFrame based on the condition
filtered_ret1 = df[condition]

# Lets filter also for the other dataframe with the returnings 
values_to_filter = filtered_ret1.index.values

filtered_data = data[data.index.isin(values_to_filter)]

#Voy a invertir $900,000 al inicio para poder tener CASH y ser capaz de hacer rebalanceos ir haciendo rebalanceos mes a mes

filtered_data.insert(1, 'Value', filtered_data['Peso (%)']*900000)
filtered_data.insert(2, 'Stocks', filtered_data['Value']/filtered_data.iloc[:,3])
stocks = filtered_data.iloc[:, [1, 2]]

#Hay que obtener los valores de nuestro portafolio para el primer periodo

df_activa.insert(1, 'capital',  filtered_data['Value'].sum())
df_activa.insert(2, 'rendimiento', 0)
df_activa.insert(3, 'rendimeinto acumulado', 0)
df_activa.insert(4, 'titulos totales', filtered_data['Stocks'].sum())
df_activa.insert(5, 'titulos_c_v', 0)
df_activa.insert(6, 'comision', filtered_data['Value'].sum()*0.004) #es la comision que cobra Kuspit
df_activa.insert(7, 'comision acumulada', filtered_data['Value'].sum()*0.04)


#%% Este es loop que cree para que cada mes se hiciera el rebalnceo, creo que quedo muy bien jeje, que orgullo

for j in range(0,24):  
    for i in range(0,9):
        filtered_data.iloc[i,1] = filtered_data.iloc[i,1]*(1+filtered_ret1.iloc[i,j+1]) #Para actualizar cuanto vale la inversion del portafolio en base a moviemientos del mercado
        # Hasta esto momento solo se ve reflajado en filtered_data el rendimiento de mercado, falta sumarle lo que compre o vendi de acciones por que los cambios en el precio (rebalanceo)

        #Para los balanceos de acciones (Aqui cambiamos la ponderacion de cada activo en el portafolio)
        if filtered_ret1.iloc[i,j+1] < -0.05:  #Row,Column
            filtered_data.iloc[i,0]= filtered_data.iloc[i,0]-0.025

        elif filtered_ret1.iloc[i,j+1] > 0.05:  
            filtered_data.iloc[i,0]= filtered_data.iloc[i,0]+0.025
        else:
            filtered_data.iloc[i,0]= filtered_data.iloc[i,0]

        #Para las compras de acciones y con ello cabiar el valor de mi portafolio
        if filtered_ret1.iloc[i,j+1] < -0.05:  #Row,Column
            filtered_data.iloc[i,1] = filtered_data.iloc[i,1]*(1-0.025)

        elif filtered_ret1.iloc[i,j+1] > 0.05:  
            filtered_data.iloc[i,1] = filtered_data.iloc[i,1]*(1+0.025)
        else:
            filtered_data.iloc[i,1] = filtered_data.iloc[i,1]*(1-0.025)

        
        
        filtered_data.iloc[i,2] = filtered_data.iloc[i,1]/filtered_data.iloc[i,j+3] #Para numero de stocks 


    #Aqui vamos a poner todos los cambios de las variables que se hacen mes a mes por el rebalanceo    
    df_activa.iloc[j+1,1]=filtered_data.Value.sum()
    df_activa.iloc[j+1,2]=(filtered_ret1.iloc[:,j+1]*filtered_ret1.iloc[:,j]).sum()
    df_activa.iloc[j+1,3]=(1+df_activa.iloc[j+1,2])*(1+1+df_activa.iloc[j,2])-1
    df_activa.iloc[j+1,4]=filtered_data.Stocks.sum()
    df_activa.iloc[j+1,5]=df_activa.iloc[j+1,4]-df_activa.iloc[j,4]
    if  df_activa.iloc[j+1,5]>0:
         df_activa.iloc[j+1,6]=df_activa.iloc[j+1,5]*0.004 #La comison que cobra kuspit
    else:
        df_activa.iloc[j+1,6]=df_activa.iloc[j+1,5]*0.004*-1
    df_activa.iloc[j+1,7]=df_activa.iloc[j,7]+df_activa.iloc[j+1,6]
    
df_activa = df_activa.drop(df_activa.index[-1])

#Solo nos queda presentar
df_activa

