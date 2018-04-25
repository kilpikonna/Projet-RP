# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:43:58 2018




Things to do :
    are prob and cumprob useful ?
    how to choose M value ?
    run tests 
    check about randomInit
    use heuristic init
    new population (2 methods)
    other selection ? crossover ? mutation ?
    
    
"""

import random
import numpy as np
from readFile import *
from UnionFind import *
import operator
from heuricticFunctions import *


"""
"""
def setOfArcsOnSubset(adjacencyMatrix, nodesSubset):
    arcs = dict()
    
    for i in range (len(nodesSubset)):
        for j in range(i+1,len(nodesSubset)):
#            print("nodeSubseeeeeeeet ",nodesSubset[i])
            if adjacencyMatrix[nodesSubset[i]][nodesSubset[j]] != 0 and adjacencyMatrix[nodesSubset[i]][nodesSubset[j]] != np.inf:
                arcs[(nodesSubset[i],nodesSubset[j])] = adjacencyMatrix[nodesSubset[i]][nodesSubset[j]]

    return arcs

"""
"""
def KruskalOnSubset(adjacencyMatrix, nodesSubset):
    UF = UnionFind(adjacencyMatrix)
    
    for node in nodesSubset:
        UF.makeSet(node)

    arcs = setOfArcsOnSubset(adjacencyMatrix,nodesSubset)
    
    arcs = sorted(arcs.items(),key = operator.itemgetter(1))
    
    print("arcs : ",arcs," and nodes : ",nodesSubset)    
    keys = [i[0] for i in arcs]
    values = [i[1] for i in arcs]
    spanningTree = dict()
    
    for i in range(len(arcs)):
        
        if UF.find(keys[i][0]) != UF.find(keys[i][1]):
            spanningTree[keys[i]] = values[i] 
            UF.union(keys[i][0], keys[i][1])
    
    return spanningTree

"""
@param : individualSize : the size ofthe individuals to generate

@param : populationSize : the number of individuals to generate

@return : a population of populationSize randomly generated individuals 
"""

def createRandomInitPopulation(individualSize, populationSize):
    
    population = []
    
    for i in range(0, populationSize) :
        p = random.uniform(0.2, 0.5)
        individual = np.zeros((individualSize))
        for j in range(0, individualSize) :
            individual[j] = np.random.choice(np.arange(2), p=[1-p, p])
        population.append(individual)
    
        print(p," population[i] ===", population[i])
        
    return population
    
"""
@param : terminals : 

@param : i : non-terminal node which index must be determined

@return : the index of the non-terminal node i in the adjacencyMatrix 
        for ex if we got 5 nodes from 0 to 4 : [0,1,2,3,4] and the terminal nodes are [0,1] and the individual [1,0,0]
        the first node of the individual refers to the node 2 but its index in the individual list is 0. Here we
        return its real index : 2

"""

def calculateRealIndex(terminals, i) :
    terminals.sort()
    index = i
    for terminal in terminals :
        if terminal <= i :
            index+=1
        else :
            
            break
    
    print("i==",i," but real index =",index)
    return index

def solutionToIndividual(solution, terminals, individualSize) :
    print("individual Size is :::::::::::::::",individualSize)
    nonTerminal_Nodes = set([])
    for arc in solution :
        if arc[0] not in terminals :
            nonTerminal_Nodes.add(arc[0])
        if arc[1] not in terminals :
            nonTerminal_Nodes.add(arc[1])

    individual = []
    
    for i in range(individualSize) :
        if calculateRealIndex(terminals, i) in nonTerminal_Nodes :
            individual.append(1)
        else : 
            individual.append(0)
    return individual

def createHeuristicInitPopulation(adjacencyMatrix, terminals, populationSize):
    
    population = []
    for i in range(populationSize) :
        newAdjacencyMatrix = inputGraphRandomization(adjacencyMatrix, 5, 20)
        solution = ACMHeuristic(newAdjacencyMatrix, terminals)
        individual = solutionToIndividual(solution.keys(), terminals, len(adjacencyMatrix)-len(terminals))
        population.append(individual)
        print(">new individual :::: ", individual, "\n>from solution : ", solution.keys()," \n>terminals : ", terminals)#, " type is :::: ", individual.shape)
    
    print("population :::::", population)
    return population

"""
@param : terminals : 

@param : individual : 

@return : the list of the nodes of the subGraph that contains the terminals and the nodes with value "1" in the individual

"""
def generateSubgraph(terminals, individual, adjacencyMatrix):
    
    subGraphNodes = []
    for terminal  in terminals :
        subGraphNodes.append(int(terminal))
    
    for i in range(0, len(individual)) :
        if individual[i]==1 : # Steiner node
            
            realIndex = calculateRealIndex(terminals, i)
            print(subGraphNodes," and the node is : ", i," but the real value is : ",realIndex) 
            subGraphNodes.append(int(realIndex))
    
    print("final subGraphNodes : ", subGraphNodes)
    return subGraphNodes

"""
@param : adjacencyMatrix : 

@param : nodesSubset : 
    
@param : M :
    
@return : the fitness of a tree/forest containing the nodes from nodesSubset list

"""    
def kruskal_fitness(adjacencyMatrix, nodesSubset, M) :
    kruskal_trees = KruskalOnSubset(adjacencyMatrix, nodesSubset)
    fitness = 0
    print("Tree : ",kruskal_trees)    
    for arc in kruskal_trees.keys() :
        fitness = fitness + kruskal_trees.get(arc)
    
    nbNodes = len(nodesSubset)
    nbEdges = len(kruskal_trees)
    if nbEdges == nbNodes-1 :# Tree    
        return fitness
    else :
        return fitness + M * (nbNodes - 1 - nbEdges)


"""
@param : adjacencyMatrix : 

@param : terminals : 
    
@param : individual :

@param : M :
    
@return : the fitness of a solution with the nodes from terminals+individual

"""  

def fitness(terminals, individual, adjacencyMatrix, M):
    nodesSubset = generateSubgraph(terminals, individual, adjacencyMatrix)
#    print(nodesSubset)    
    weights = kruskal_fitness(adjacencyMatrix, nodesSubset, M)   
    return weights
   

"""
@param : fitnessList : dictionary containing ...

    
@return : two parents selected proportionally to their fitness using cumulative prob

"""      
def rouletteWheelSelection(fitnessList) : # select parents
        
    # select a random a number in the range [0,1]
    random_num1=random.random()
    random_num2=random.random()
    parent1 = None
    parent2 = None
    
    for individual in fitnessList.keys() :
        if fitnessList[individual][3] > random_num1 and parent1 == None :
            parent1 = individual
        if fitnessList[individual][3] > random_num2 and parent2 == None :    
            parent2 = individual
        if parent1 != None and parent2 != None :
            break
        
    return parent1, parent2

#def parentSelection(fitnessList, selectionType) :
#    return None

"""
@param : parent1

@param : parent2
    
@return : performs a one point crossover : swaps the tails from the two parents (swap index generated randomly) and returns 2 children

"""  
def onePoint_crossover(parent1, parent2) :
    index = random.randint(0, len(parent1)-1)
    
    child1 = [parent1[i] for i in range(index)]
    child1.extend(parent2[i] for i in range(index, len(parent2)))
    
    child2 = [parent2[i] for i in range(index)]
    child2.extend(parent1[i] for i in range(index, len(parent1)))
    
    print("-----------------------------------", child1)
    return child1, child2
    

"""
@param : child :
    
@return : selects a random node and reverses its value (add it to the solution if wasn't considered or delets it)

"""  
def bitFlip_mutation(child) : 
    perform_mutation = random.random()
    if(perform_mutation<=1/10) :
        index = random.randint(0, len(child)-1)
        child[index] = (child[index]+1)%2


"""
@param : adjacencyMatrix : 

@param : terminals : 

@param : n :
    
@param : M :
    
@return : the best solution found with genetic algorithm

"""  
def simpleGeneticAlgorithm(adjacencyMatrix, terminals, n, M) :
    
#    population = createRandomInitPopulation(len(adjacencyMatrix)-len(terminals), n)
    population = createHeuristicInitPopulation(matrix, [0,1], n)
    
    for k in range(n) :
        
        print("********************************* k=",k)
        print("*************population : ",population)
        fitnessList = dict()
        fitnessSum = 0
        
        # calculate and adapts fitness to fit the rule : the more the fit the more the individual's prob is
        for i in range(n) :
            individual = tuple(population[i])
            print("populasse : ",population[i])
            fit = fitness(terminals, individual, adjacencyMatrix, M)
            fitnessList[individual] = [fit, 1/fit, 0, 0] # {individual : [fitness, new fitness, probability, cumprob]}
            fitnessSum += fitnessList[individual][1]
        # calculates the probability of each individual + cumulative prob
        cumprob = 0
        for i in range(n) :
            individual = tuple(population[i])
            prob = fitnessList[individual][1]/fitnessSum
            fitnessList[individual][2] = prob
            cumprob +=  prob
            fitnessList[individual][3] = cumprob
            
        population = []
            
        for i in range(int(n/2)) :
            # selection : select two parents (individuals) among the population 
            parent1, parent2 = rouletteWheelSelection(fitnessList)              
            # crossover : we select a random crossover point and exchange the tails of the parents to get 2 new individuals
            child1, child2 = onePoint_crossover(parent1, parent2)
            # mutation : exchange the value of a random node
            bitFlip_mutation(child1)
            bitFlip_mutation(child2)
            # insertion : insert new individuals to the population
            population.append(child1)
            population.append(child2)
       
        
           
        
#        population = set(population) # is a good idea ?
        
        
    bestIndividual = []
    for individual in fitnessList.keys() :
#        individual = fitnessList[i]
        if len(bestIndividual) == 0 or bestIndividual[1] > fitnessList[individual][0] :
            bestIndividual = [individual, fitnessList[individual][0]]
    
    return bestIndividual
        

#adjacencyMatrix, terminals = readInstance(os.getcwd()+"\heuristic\instance039.gr")

M = 100000*100000
n = 5

#print(adjacencyMatrix, terminals)
#print(simpleGeneticAlgorithm(adjacencyMatrix, terminals, n, M))


matrix = np.zeros((5,5))
matrix[0] = [0,0,2,1,5]
matrix[1] = [0,0,6,4,2]
matrix[2] = [2,6,0,3,1]
matrix[3] = [1,4,3,0,2]
matrix[4] = [5,2,1,2,0]
#parent1 = [0,0,1,1,0]
#parent2 = [0,0,0,1,1]
n=4

print(createHeuristicInitPopulation(matrix, [0,1], 10))

print(simpleGeneticAlgorithm(matrix, [0,1], n, M))

