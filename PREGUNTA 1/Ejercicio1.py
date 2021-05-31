# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:12:14 2021

@author: Gabriel Condori
"""

import pandas as pd
import numpy as np
import random


def leer_archivo():
    global matriz
    global titulos
   
    #Leemos el dataset con las distancias
    df = pd.read_csv("distancias.csv")
    matriz=df.to_numpy()
    titulos=list(df.columns)
    #print(len(matriz),len(matriz[1]))
    for i in range(len(titulos)):
        indices[titulos[i]]=i
    global max
    
    max=len(titulos)
    #print(indices)
    print(matriz)
    #print(titulos)
def individual(nodo):
    vec=list(range(min,max))
    vec.remove(nodo)
    ind=[]
    ind.append(nodo)
    for i in range(max-1):
        aux=random.sample(vec, 1)[0]
        ind.append(aux)
        vec.remove(aux)
    ind.append(nodo)
    return ind
    
def newPopulation(nodo):
    return [individual(nodo) for i in range(num)]
def functionType(individual):
    fitness = 0
    for i in range(len(individual)-1):
        
        fitness+=matriz[individual[i]][individual[i+1]]
    return fitness
def selection_and_reproduction(population):
    evaluating = [ (functionType(i), i) for i in population]
    #print("eval",evaluating)
    evaluating = [i[1] for i in sorted(evaluating,reverse=True)]
    #print("eval",evaluating)
    population = evaluating
    selected = evaluating[(len(evaluating)-pressure):]
    
    tamano=len(population)
    cont=0
    i=-1
    pop_aux=[]
    while(cont<tamano):
        i+=1
        if(len(population)-pressure==i):
            i=0
        pointChange = random.randint(1,max-1)
        father = random.sample(selected, 2)
        population[i][:pointChange] = father[0][:pointChange]
        population[i][pointChange:] = father[1][pointChange:]  
        if len(set(population[i]))==max:
            pop_aux.append(population[i].copy())
            cont+=1
            
    return pop_aux 

'''def selection_and_reproduction1(population):
    evaluating = [ (functionType(i), i) for i in population]
    print("eval",evaluating)
    evaluating = [i[1] for i in sorted(evaluating,reverse=True)]
    print("eval",evaluating)
    population = evaluating
    selected = evaluating[(len(evaluating)-pressure):]
    
    
    for i in range(len(population)-pressure):
        pointChange = random.randint(1,max-1)
        father = random.sample(selected, 2)
        print('-------------------------------------------------')
        print(father)
        print('-------------------------------------------------')
        population[i][:pointChange] = father[0][:pointChange]
        population[i][pointChange:] = father[1][pointChange:]       
    return population
'''
# num = numero de poblacion
# pressure = numero de muestra
# Generation = numero de generaciones
num=100
min=0
indices=dict()
leer_archivo()
nodo=input('Ingrese el nodo inicial: ').upper()
poblacion=(newPopulation(indices[nodo]))
pressure=2
generation=100
#print(np.array(poblacion))

#print("\Selection Population:\n%s"%(np.array(poblacion)))
# Iteramos todas las generaciones
for i in range(generation):
    poblacion = selection_and_reproduction(poblacion)
# Mostramos la solucion encontrada
#print(np.array(poblacion),functionType(poblacion[1]))
print('Mejor camino a partir del nodo (mejor indiviuo)',nodo,': ')
print("Mejor individuo: %s" % [list(indices.keys())[list(indices.values()).index(i)]for i in poblacion[0]] )

for i in poblacion[0]:
    print(list(indices.keys())[list(indices.values()).index(i)],'--->',i)
print('Distancia Minima a recorrer: ',functionType(poblacion[0]))
#poblacion = mutation(population)
#print(functionType(poblacion[0]))
