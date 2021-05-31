# -*- coding: utf-8 -*-
"""
Created on Tue May 25 00:09:11 2021

@author: IBM GAMER
"""

import pandas as pd
import numpy as np
"""
Para resolver este problema se empleará el algoritmo del vecino mas cercano, ya que este algoritmo era
utilizado para poder dar solucion al preblema del Agente Viajero.
"""
def DistanciaCamino(camino, matrizDistancias):
    """
    Recibirá el recorrido del camino y la matriz de distancias, el metodo nos devolverá el recorrido o distancia
    menor del camino evaluado
    """
    distancia = sum(matrizDistancias[camino[i]][camino[i+1]] 
        if camino[i] >= camino[i+1]
        else matrizDistancias[camino[i+1]][camino[i]]
        for i in camino[:-1])
    return distancia

def NNA(nodoInicial, matrizDistancias):
    """
    A continuacion se muestra el desarrollo del Vecino más Cercano (Nearest Neighbor Algotithm).
    Este algoritmo necesita un nodo inicial y la matriz de distancias para poder determinar el camino más óptimo.
    Obtendrá el camino más corto pero no el más ideal.
    """
    # Elección de un vértice arbitrario respecto al vértice actual.
    candidatos = list(range(len(matrizDistancias)))
    candidatos.remove(nodoInicial)
    estado = nodoInicial
    camino = [estado]
    # Bucle de búsqueda
    while True:
        if len(candidatos) == 0:
            camino.append(nodoInicial)
            break # Cierre cuando no existen mas candidatos
        # Selecciona el candidato con el menor recorrido o distancia
        estado = min(candidatos, 
                     key=(lambda j: matrizDistancias[estado][j] 
                          if estado >= j else matrizDistancias[j][estado]))
        # Agrega el estado al camino y lo remueve de la lista de candidatos
        camino.append(estado), candidatos.remove(estado)
    return camino
def main():
    #Leemos el dataset con las distancias
    df = pd.read_csv("distancias.csv")
    print(df)
    #Convirtiendo el df a matriz
    matriz = np.array(df)
    global max
    max=len(matriz)
    av = {
      "Cant_Lugares" : len(matriz),#Cantidad de lugares que se visitará
      "Matriz_Distancias" : matriz  
    }
    distancias = av["Matriz_Distancias"]
    #le mandamos el estado de donde comenzara el tour
    camino = NNA(0, distancias)
    fitness = DistanciaCamino(camino, distancias)
    return camino, fitness
if __name__ == '__main__':
    camino, distanciaMin = main()
    print("Distancia Minima a recorrer: ", distanciaMin) 
    diccionario=dict()
    for i in range(max):
        diccionario[i]=chr(65+i)
    print("Mejor individuo (camino):", camino)
    
    print("Mejor individuo (camino):", [diccionario[i] for i in camino])
    
