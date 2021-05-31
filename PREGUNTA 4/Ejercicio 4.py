# -*- coding: utf-8 -*-
"""

@author: Gabriel Condori
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def X_Entrenamiento(X, n):
    X_T=[]
    for i in range(int(n*0.8)):#80% de la data
        X_T.append(X[i])
    X_T = np.array(X_T)
    return(X_T)
def X_Evaluacion(X, n):
    X_T=[]
    for i in range(int(n*0.8),n):#20% de la data
        X_T.append(X[i])
    X_T = np.array(X_T)
    return(X_T)
def y_Entrenamiento(y, n):
    y_T=[]
    for i in range(int(n*0.8)):#80% de la data
        y_T.append(y[i])
    return(y_T)
def y_Evaluacion(y, n):
    y_T=[]
    for i in range(int(n*0.8),n):#20% de la data
        y_T.append(y[i])
    return(y_T)


#LECTURA DEL DATASET
df = pd.read_csv("audit_risk.csv")
n = len(df.index)#Cantidad de filas dentro del dataset
#PREPROCESAMIENTO
print(df.dtypes)
"""
Se puede ver en la descripcion de las columnas obtenidas de la pagina que 6 de 
las 16 columnas son datos continuos, pero al momento de la importacion pandas
cambio el tipo de dato de las columnas A2 y A14, por ende se realizará la
conversion correspondiente.
"""
#df["LOCATION_ID"] = pd.to_numeric(df["LOCATION_ID"], errors='coerce')
#df["A14"] = pd.to_numeric(df["A14"], errors='coerce')
"""
Una vez realizado el proceso se tiene el resultado original de los tipos de datos
de las columnas. Con esta conversion aparecerán los datos NaN que se corregiran
más adelante.
"""
print('--------------------------------------')
print(df.dtypes)

#Se obtendrán las columnas numericas y de objetos en dos df diferentes
dataNum = df[[
'Sector_score',
'LOCATION_ID',
'PARA_A',
'Score_A',
'Risk_A',
'PARA_B',
'Score_B',
'Risk_B',
'TOTAL',
'numbers',
'Score_B.1',
'Risk_C',
'Money_Value',
'Score_MV',
'Risk_D',
'District_Loss',
'PROB',
'RiSk_E',
'History',
'Prob',
'Risk_F',
'Score',
'Inherent_Risk',
'CONTROL_RISK',
'Detection_Risk',
'Audit_Risk',
'Risk']]
              

dataObj = df[['LOCATION_ID']]
print(dataNum)
print(dataObj)
#Realizando imputacion al df numerico
imputacionN = SimpleImputer(missing_values=np.nan,strategy="mean")
dataNumImp = imputacionN.fit_transform(dataNum)
print(dataNumImp)
"""
Se creara un nuevo dataset a partir del nuevo dataset preprocesado y con el cual
se implementara en MLP
"""
elementos={
    "A1": dataNumImp[:,0],
    "A2": dataNumImp[:,1],
    "A3": dataNumImp[:,1],
    "A4": dataNumImp[:,2],
    "A5": dataNumImp[:,3],
    "A6": dataNumImp[:,4],
    "A7": dataNumImp[:,5],
    "A8": dataNumImp[:,6],
    "A9": dataNumImp[:,7],
    "A10": dataNumImp[:,8],
    "A11": dataNumImp[:,9],
    "A12": dataNumImp[:,10],
    "A13": dataNumImp[:,11],
    "A14": dataNumImp[:,12],
    "A15": dataNumImp[:,13],
    "A16": dataNumImp[:,14],
    "A17": dataNumImp[:,15],
    "A18": dataNumImp[:,18]
    
    
}
data = pd.DataFrame(elementos)
print(data)


#Implementacion del MLPClassifier
X = np.array(data[["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11",
                   "A12","A13","A14","A15","A16","A17","A18"]])#Data


X_TR = X_Entrenamiento(X,n)#Data de Entrenamiento
X_TS = X_Evaluacion(X,n)#Data de Evaluacion

y = np.array(data["A2"])#Objetivos

y_TR = y_Entrenamiento(y, n)#Objetivos de Entrenamiento
y_TS = y_Evaluacion(y, n)#Objetivos de Evaluacion
print("Data de entrenamiento")
print(X_TR)
print("Objetivo de entrenamiento")
print(y_TR)
print("Data de testeo")
print(X_TR)
print("Objetivo de testeo")
print(y_TS)
clasificador = MLPClassifier(tol=1e-2)
clasificador.fit(X_TR, y_TR)
y_PR = clasificador.predict(X_TS)#Objetivos Predecidos con la data de Evaluacion
print("PREDICCION")
print(y_PR)
cm = confusion_matrix(y_TS, y_PR)#Comparacion entre los Objetivos de Testeo y los Objetivos Predecidos
print("MATRIZ DE CONFUSION")
print(cm)