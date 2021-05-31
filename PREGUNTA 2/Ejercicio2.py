# -*- coding: utf-8 -*-
"""
Created on Tue May 25 00:14:03 2021

@author: IBM GAMER
"""

"""**Importamos las liberías necesarias:**
1. random: números pseudoaleatorios
2. numpy: arrays
3. maplotlib.pyplot: visualizar los resultados
4. deap.base: incluye las clases base de deap. En concreto dos son importantes en nuestro ejemplo, base.Fitness y base.Toolbox.
5. deap.creator: permite crear clases nuevas.
6. deap.tools: herramientas para implementar los algoritmos genéticos: operadores genéticos (selección, cruce y mutación), hallofFame, estadística, registro de evolución, etc.
7. deap.algorihtms: incluye implementaciones completas de algoritmos genéticos, nosotros vamos a utilizar eaSimple.
"""

import random
import pandas as pd
import numpy

import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

indices=dict()
def leer_archivo():
    global matriz
    global titulos
   
    #Leemos el dataset con las distancias
    df = pd.read_csv("distancias.csv")
    matriz=df.to_numpy()
    titulos=list(df.columns)
    for i in range(len(titulos)):
        indices[titulos[i]]=i
    global max
    
    max=len(titulos)
    print(matriz)

"""**Diccionario que contiene los datos del problema:**
1. Toursize: número de ciudades.
2. OptTour: tour óptimo para este problema. (este dato no es conocido normalmente)
3. OptDistance: distancia recorrida por el agente para el tour óptimo.
4. DistanceMatrix: matriz de distancia.

tsp = {
"TourSize" : 5,
"OptTour" : [3,2,1,0,4],
"OptDistance" : 0,
"DistanceMatrix" :
    [[ 0, 7,  9,  8, 20],
     [ 7,  0, 10,  4, 11],
     [ 9, 10,  0, 15,  5],
    [ 8,  4, 15,  0, 17],
    [20, 11,  5, 17,  0]]
}
"""

leer_archivo()
tsp = {
"TourSize" : 5,
"OptTour" : [3,2,1,0,4],
"OptDistance" : 0,
"DistanceMatrix" :matriz
}

"""Guardamos en dos variables distintas la matriz de distancia y el número de ciudades. 
Accedemos a los valores mediante las claves del diccionario."""

distance_map = tsp["DistanceMatrix"] 
IND_SIZE = tsp["TourSize"]

"""Creamos la clase que define el fitness de los individuos **FitnessMin**. Este paso en la mayoría 
de los problemas será muy parecido. Siempre tendremos que heredar de base.Fitness. El atributo 
**weights** nos dice el número de objetivos de nuestro problema y el tipo (-1.0 para minimizar y 
1.0 para maximizar). En este caso es un problema de **minimización** de **un objetivo**. **En deap 
el caso mono objetivo es un caso particula del multiobjetivo**."""

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

"""Ahora creamos la "plantilla" del individuo, **el cromosoma**. El individuo **será una lista** 
(hereda los métodos de la lista), pero tiene el atributo **FitnessMin** creado en la línea anterior. 
Representar los individuos como lista nos servirá en una gran cantidad de casos (lista de variables 
del problema)."""

creator.create("Individual", list, fitness=creator.FitnessMin)

"""El objeto toolbox funciona como una "caja de herramientas" donde **debemos registrar operaciones 
que nos hacen falta en el algoritmo genético**. Cosas que debemos registrar:

1. Funciones para crear tanto individuos aleatorios como la población inicial.
2. Operadores genéticos (selección, cruce y mutación).
3. La función de evaluación.
"""

toolbox = base.Toolbox()

"""Comenzamos registrando las funciones que nos permiten generar individuos aleatorios y la población 
incial. Empezamos por muestra aleatorias de individuos (cromosoma). **No es el individuo completo ya que no tiene el  atributo fitness**."""

# Generación de un tour aleatorio
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE) # aquí debemos registar una función que generar una muestra de individuo

"""Podemos ver la muestra que se crea:"""

print(toolbox.indices())

"""Generamos indivuos aleatorios y población inicial"""

# Generación de inviduos y población
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
POP_SIZE=100
toolbox.register("population", tools.initRepeat, list, toolbox.individual, POP_SIZE) #

"""Vamos a ver cómo funcionan con un par de ejemplos:"""

ind = toolbox. individual() # creamos un individuo aleatorio
print(ind)
print(ind.fitness.values) # en fitness.values se guardará el fitness

pop = toolbox.population() # creamos una población aleatoria
print(pop[:5]) # imprimimos los 5 primeros individuos

"""**Definimos la función objetivo:**
1.   En primer lugar calculamos la distancia entre la última ciudad y la primera 
2.   Recorremos dos listas la vez:
*   individual[0:-1], son los elementos del primero al penúltimo
*   individual[1:], son los elementos del segundo al último
Por lo tanto en cada interación, calculamos la distancia entre dos ciudades consecutivas.
**IMPORTANTE:** Siempre debemos devolver una tupla!!
"""

def evalTSP(individual):
    """ Función objetivo, calcula la distancia que recorre el viajante"""
    # distancia entre el último elemento y el primero
    distance = distance_map[individual[-1]][individual[0]]
    # distancia entre el resto de ciudades
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return distance,

"""**Registro de operadores genéticos:**
1.   Cruce ordenado.
2.   Mutación mediante mezcla de índices.
3.   Selección mediante torneo, tamaño del torneo igual a 3 (suele ir bien para la mayoría de 
     problemas). Si tenemos muchas variables podemos probar a aumentarlo un poco.
4.   Función de evaluación.
"""

toolbox.register("mate", tools.cxOrdered)                       
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05) 
toolbox.register("select", tools.selTournament, tournsize=3)    
toolbox.register("evaluate", evalTSP)

"""Función que nos permite visualizar la evolución del algoritmo. Recibe como entrada el registro 
de evolución."""

def plot_evolucion(log):
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")

    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    ax1.plot(gen, fit_maxs, "r")
    ax1.plot(gen, fit_ave, "--k")
    ax1.fill_between(gen, fit_mins, fit_maxs, 
                     where=fit_maxs >= fit_mins, 
                     facecolor="g", alpha=0.2)
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Fitness")
    ax1.legend(["Min", "Max", "Avg"])
    ax1.set_ylim([2000, 6000])
    plt.grid(True)
    
"""En el main configuramos el algoritmo genético. 
Ajuste de los operadores genéticos:
* *CXPB:* probabilidad de cruce
* *MUTPB:* probabilidad de mutación
Número de generaciones:
* *NGEN:* número de generaciones
El objeto **hof** almacena el mejor individuo encontrado a lo largo de las generaciones. Le tenemos que p
El objeto **stats** calcula las estadísticas de la población en cada generación. Cuando se define le tenemos que decir sobre qué se va a calcular las estadística. A continuación, se deben registrar las funciones estadísticas que se van a aplicar.
El objeto **logbook** almacena todas las estadísticas calculadas por generación en un solo objeto.
Algoritmo **eaSimple** parámetros que tenemos que pasar:
* poblacion (obligatorio)
* toolbox (obligatorio)
* probabilidad de cruce (obligatorio)
* probabilidad de mutación (obligatorio)
* número de generaciones (obligatorio)
* objeto de estadísticas (opcional)
* objeto hallofFame que almacena el mejor individuo (opcional)
* Si queremos que se muestre las estádisticas en cada generación, verbose = True (opcional)
"""

def main():
    random.seed(9) # ajuste de la semilla del generador de números aleatorios
    CXPB, MUTPB, NGEN = 0.7, 0.3, 120
    pop = toolbox.population() # creamos la población inicial 
    hof = tools.HallOfFame(1) 
    stats = tools.Statistics(lambda ind: ind.fitness.values) 
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()     
    pop, logbook = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=True)
    return hof, logbook

"""Lanzamos el algoritmo y mostramos los resultados."""

hof, log = main()
print(log)
print("Mejor fitness (camino mas corto): %f" %hof[0].fitness.values)
print("Mejor individuo: %s" %hof[0])
print("Mejor individuo: %s" % [list(indices.keys())[list(indices.values()).index(i)]for i in hof[0]] )
for i in hof[0]:
    print(list(indices.keys())[list(indices.values()).index(i)],'--->',i)
print("\n\n\n\n\n\n")
plot_evolucion(log) # mostamos la evolución