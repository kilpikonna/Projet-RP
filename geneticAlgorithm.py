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
import time


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
    
    #print("arcs : ",arcs," and nodes : ",nodesSubset)    
    keys = [i[0] for i in arcs]
    values = [i[1] for i in arcs]
    spanningTree = dict()
    
    for i in range(len(arcs)):
        
        if UF.find(keys[i][0]) != UF.find(keys[i][1]):
            spanningTree[keys[i]] = values[i] 
            UF.union(keys[i][0], keys[i][1])
    
    return spanningTree

"""
Part 1.6 of the Project

@param : individualSize : the size of the individuals to generate

@param : populationSize : the number of individuals to generate

@return : list() of np.array() : a population of populationSize randomly generated individuals 
"""

def createRandomInitPopulation(individualSize, populationSize):
    
    population = []
    
    for i in range(0, populationSize) :
        p = random.uniform(0.2, 0.5)
        individual = np.zeros((individualSize))
        for j in range(0, individualSize) :
            individual[j] = np.random.choice(np.arange(2), p=[1-p, p])
        population.append(individual)
        nb = [i for i in individual].count(0)
#        print("number of zeros : ", nb)
    
    return population
    
"""
@param : terminals : list() the list containing the index of terminal nodes

@param : index : index of the non-terminal node in the individual coding

@return : the index of the non-terminal node i in the adjacencyMatrix 
        for ex if we got 5 nodes from 0 to 4 : [0,1,2,3,4] and the terminal nodes are [0,1] and the individual [1,0,0]
        the first node of the individual refers to the node 2 but its index in the individual list is 0. Here we
        return its real index : 2

"""

def calculateRealIndex(terminals, index) :
    terminals.sort()
#    print(terminals)
    i = index
#    index = i
    for terminal in terminals :
        if terminal <= index :
#            print("terminal : ",terminal," is lower than ",index)
            index+=1
        else :            
            break
    
#    print("i==",i," but real index =",index)
    return index
"""
@param : solution : dict() : contains a set of arcs of the spanning tree of a solution to the Steiner problem

@param : terminals : list() the list containing the index of terminal nodes

@param : individualSize : the number of non-terminal nodes 

@return : list() : result of the conversion of a solution as returned by the heuristic functions to an individual 
                    as coded here (as described in Part 1.1 of the Project)

"""
def solutionToIndividual(solution, terminals, individualSize) :
    #print("individual Size is :::::::::::::::",individualSize)
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

"""
Part 2.3 of the Project

@param : adjacencyMatrix : np.array(float, float), the complete adjacency matrix of graph, adjacencyMatrix[i][j] contains the value of the arc connecting 
                        nodes i and j

@param : terminals :  list() the list containing the index of terminal nodes

@param : populationSize : the number of the individuals in the population considered
    
@return : list() of list() : a list of individuals generated using the heuristics as described in the part 2 of the Project
                            applied on a perturbed adjacencyMatrix
"""

def createHeuristicInitPopulation(adjacencyMatrix, terminals, populationSize):
    
    population = []
#    for i in range(int(populationSize/2)) :
    for i in range(int(populationSize)) :
        #Perturbation of the data of the adjacencyMatrix
        newAdjacencyMatrix = inputGraphRandomization(adjacencyMatrix, 5, 20)
        
        #Solution generated using the spanning tree heuristic (Part 2.2)
        solution = ACMHeuristic(newAdjacencyMatrix, terminals)
        individual = solutionToIndividual(solution.keys(), terminals, len(adjacencyMatrix)-len(terminals))
        population.append(individual)
        
        #Solution generated using the shortest path heuristic (Part 2.1) TODO!!!!!!!!!!!!!!
#        solution = distanceGraphHeuristic(newAdjacencyMatrix, terminals)
#        individual = solutionToIndividual(solution.keys(), terminals, len(adjacencyMatrix)-len(terminals))
#        population.append(individual)
        
        #print(">new individual :::: ", individual, "\n>from solution : ", solution.keys()," \n>terminals : ", terminals)#, " type is :::: ", individual.shape)
    
    #print("population :::::", population)
    return population

"""
@param : terminals : list() the list containing the index of terminal nodes

@param : individual : tuple() : an individual representing the Steiner nodes (1 if it is, 0 if it is not)

@return : list() of index : the list of the nodes of the subGraph that contains the terminals and the Steiner nodes
                                (the nodes with value "1" in the individual coding)

"""
def generateSubgraph(terminals, individual, adjacencyMatrix):
    
    subGraphNodes = []
    for terminal  in terminals :
#        print(terminal)
        subGraphNodes.append(int(terminal))
#        print("subGraph", subGraphNodes)
    
    for i in range(0, len(individual)) :
        if individual[i]==1 : # Steiner node           
            realIndex = calculateRealIndex(terminals, i)
            #print(subGraphNodes," and the node is : ", i," but the real value is : ",realIndex) 
            subGraphNodes.append(int(realIndex))
    
    #print("final subGraphNodes : ", subGraphNodes)
    return subGraphNodes

"""
Part 1.2 of the Project

@param : adjacencyMatrix : np.array(float, float), the complete adjacency matrix of graph, adjacencyMatrix[i][j] contains the value of the arc connecting 
                        nodes i and j

@param : nodesSubset : list() of index of the nodes of the subgraph considered
    
@param : M : Great number used to penalize bad solutions (the ones that don't return a tree but a forest)
    
@return : the fitness of a tree/forest containing the nodes from nodesSubset list using kruskal algorithm

"""    
def kruskal_fitness(adjacencyMatrix, nodesSubset, M) :
    kruskal_trees = KruskalOnSubset(adjacencyMatrix, nodesSubset)
    fitness = 0
    #print("Tree : ",kruskal_trees)    
    for arc in kruskal_trees.keys() :
        fitness = fitness + kruskal_trees.get(arc)
    
    nbNodes = len(nodesSubset)
    nbEdges = len(kruskal_trees)
    
    if nbEdges == nbNodes-1 :# Tree    
        return fitness
    else :
        return fitness + M * (nbNodes - 1 - nbEdges)


"""
@param : adjacencyMatrix : np.array(float, float), the complete adjacency matrix of graph, adjacencyMatrix[i][j] contains the value of the arc connecting 
                        nodes i and j

@param : terminals : list() the list containing the index of terminal nodes
    
@param : individual : tuple() : an individual representing the Steiner nodes (1 if it is, 0 if it is not)

@param : M : Great number used to penalize bad solutions (the ones that don't return a tree but a forest)
    
@return : the fitness of a solution with the nodes from terminals+individual

"""  

def fitness(terminals, individual, adjacencyMatrix, M):
    nodesSubset = generateSubgraph(terminals, individual, adjacencyMatrix)
    nodesSubset.sort(key=int)

    weights = kruskal_fitness(adjacencyMatrix, nodesSubset, M)   
    
    return weights
   

"""
Part 1.3.1 of the Project : parents selection
@param : fitnessList : list() of list() of shape [individual, fitness of this individual, 1/fitness of the individual]

@param : fitnessSum : the sum of the finess values of all the individuals of the population
    
@return : two parents of each of type tuple, selected proportionally to their fitness using the roulette wheel selection

"""      
def rouletteWheelSelection(fitnessList, fitnessSum) : # select parents
        
    # select a random a number in the range [0,1]
    random_num1=random.uniform(0,fitnessSum)#.random() #.randint(0, fitnessSum)
    random_num2=random.uniform(0,fitnessSum)#.random() #(0, fitnessSum)
    parent1 = None
    parent2 = None
    partialSum = 0
    
#    parent1 = fitnessList[0][0]
#    parent2 = fitnessList[1][0]
    
    
    for i in range(len(fitnessList)) :
        partialSum = partialSum + fitnessList[i][1]
        if partialSum >= random_num1 and parent1 == None :
            parent1 = fitnessList[i][0]
        if partialSum >= random_num2 and parent2 == None :    
            parent2 = fitnessList[i][0]
        if parent1 != None and parent2 != None :
            break
    
    return parent1, parent2


"""
Part 1.3.2 of the Project : children generation

@param : parent1 : tuple() : an individual representing the Steiner nodes (1 if it is, 0 if it is not)

@param : parent2 : tuple() : an individual representing the Steiner nodes (1 if it is, 0 if it is not)
    
@return : two children of type np.array(), generated by performing a one point crossover on the parents: 
            swaps the tails from the two parents (swap index generated randomly) and returns 2 children

"""  
def onePoint_crossover(parent1, parent2) :
    index = random.randint(0, len(parent1)-1)
    
    child1 = [parent1[i] for i in range(index)]
    child1.extend(parent2[i] for i in range(index, len(parent2)))
    
    child2 = [parent2[i] for i in range(index)]
    child2.extend(parent1[i] for i in range(index, len(parent1)))
    
    child1 = np.array(child1)
    child2 = np.array(child2)
    #print("-----------------------------------", child1)
    return child1, child2
    

"""
Part 1.4 of the Project : Mutation

performs a mutation of the individual with probability 4/100 : selects a random node and reverses its value 
(adds it to the solution if wasn't considered or deletes it if it was)

@param : child : np.array() : individual
    
"""  
def bitFlip_mutation(child) : 
    perform_mutation = random.random()
    if(perform_mutation<=4/100) :
        index = random.randint(0, len(child)-1)
        child[index] = (child[index]+1)%2


"""
@param : fitnessList : list() of list() of shape [individual, fitness of this individual, 1/fitness of the individual]

@param : populationSize : the number of individuals to generate

@return : list() of np.array() : the n=populationSize best individuals (those with the best fitness = lowest fitness)

"""
def bestIndividuals(fitnessList, populationSize) :
#    sorted_fitnessList = sorted(fitnessList.items(), key=operator.itemgetter(1))
    fitnessList.sort(key=operator.itemgetter(1))
    m = len(fitnessList)
    
    population = [np.array(fitnessList[i][0]) for i in range(min(populationSize, m))]
    
    return population


"""
@param : population : list() of individuals of type np.array() 

@return : set() of distinct individuals of that population
"""
def getCategories(population) :
    distinctIndividuals = set()
    for individual in population :
        distinctIndividuals.add(tuple(individual))
        
    return distinctIndividuals

"""
@param : population : list() of individuals of type np.array()

@param : individual : np.array()

@return : 0 if the individual is among that population and 1 otherwise

"""
def notIn(population, individual) :
    for individual2 in population :
#        print("1 :", type(individual),"  2:",type(individual2))
        if (individual2 == individual).all() :
                return 0
    return 1

"""

@param : oldPopulation : list() of individuals of type np.array()
    
@param : newPopulation : list() of individuals of type np.array()
    
@return : int : number of the new individuals of the population
"""
def getNbOfNewIndividuals(oldPopulation, newPopulation) :
    new = 0
    for individual in newPopulation :
        new += notIn(oldPopulation, individual)
    
    return new


"""
@param : fitnessList : list() of list() of shape [individual, fitness of this individual, 1/fitness of the individual]

@param : populationSize : the number of individuals to generate

@return : list() of np.array() : the n=populationSize best distinct individuals (those with the best fitness = lowest fitness)
                                

"""
def bestDistinctIndividuals(fitnessList, populationSize) :
    m = len(fitnessList)
    
    population = [fitnessList[i][0] for i in range(len(fitnessList))] #type is tuple
    
#    print("////////////////////////////////////////////",type(fitnessList[i][0]))
    populationSet = set(population)
    
#    print("set=========", populationSet,"\n")
    
    sortedPopulationSet = []
#    sortedPopulationSet[0] = 1
    for individual in populationSet :
        for fitIndividual in fitnessList :
            if fitIndividual[0] == individual :
                sortedPopulationSet.append([individual, fitIndividual[1]])
                break
    
    sortedPopulationSet.sort(key=operator.itemgetter(1))
    
#    print("sorted ::::",sortedPopulationSet)
    
    bestIndvs = [np.array(individual[0]) for individual in sortedPopulationSet]
    
#    print("best ::::",bestIndvs)
    
    if len(populationSet) < populationSize :
        for individual in population :
            bestIndvs.append(np.array(individual))
        return bestIndvs[:populationSize]
    else :        
        return bestIndvs[:populationSize]

"""
@param : fitnessList : list() of list() of shape [individual, fitness of this individual, 1/fitness of the individual]

@param : fitnessSum : the sum of the finess values of all the individuals of the population

@param : populationSize : the number of individuals to generate

@param : j : [0,2] : determines whetherthe priority is given to have a rich population (various individuals)
                        or a population of the best individuals (regardless of the duplicates)

@return : list() of np.array() : the new population of the best individuals among parents and children
"""

def elitisimBasedSurvivorSelection(fitnessList, fitnessSum, populationSize, j) :
    fitnessList.sort(key=operator.itemgetter(1))    
    for i in range(populationSize) :
        # selection : select two parents (individuals) among the population 
        parent1, parent2 = rouletteWheelSelection(fitnessList.copy(), fitnessSum)              
        # crossover : we select a random crossover point and exchange the tails of the parents to get 2 new individuals
        child1, child2 = onePoint_crossover(parent1, parent2)
        # mutation : exchange the value of a random node
        bitFlip_mutation(child1)
        bitFlip_mutation(child2)
        # insertion : insert new individuals to the population
        tchild1 = tuple(child1) 
        fit = fitness(terminals, tchild1, adjacencyMatrix, M)
        fitnessList.append([tchild1, fit, 1/fit]) #, 0, 0]
        
        tchild2 = tuple(child2) 
        fit = fitness(terminals, tchild2, adjacencyMatrix, M)
        fitnessList.append([tchild2, fit, 1/fit])#, 0, 0]
        
        
#    fitnessList.sort(key=operator.itemgetter(1))
    if j==0 or j==2 : 
        bestIndvs = bestIndividuals(fitnessList, populationSize)
    else :
        bestIndvs = bestDistinctIndividuals(fitnessList, populationSize)
    
    return bestIndvs

"""
@param : fitnessList : list() of list() of shape [individual, fitness of this individual, 1/fitness of the individual]

@param : fitnessSum : the sum of the finess values of all the individuals of the population

@param : populationSize : the number of individuals to generate

@return : list() of np.array() : the new population contains only the children

"""

def childrenBasedSurvivorSelection(fitnessList, fitnessSum, populationSize) :
    population = []
    for i in range(int(populationSize/2)) :
        # selection : select two parents (individuals) among the population 
        parent1, parent2 = rouletteWheelSelection(fitnessList.copy(), fitnessSum)              
        # crossover : we select a random crossover point and exchange the tails of the parents to get 2 new individuals
        child1, child2 = onePoint_crossover(parent1, parent2)
        # mutation : exchange the value of a random node
        bitFlip_mutation(child1)
        bitFlip_mutation(child2)
        # insertion : insert new individuals to the population
        population.append(child1)
        population.append(child2)
        
    return population
        
"""
@param : adjacencyMatrix : np.array(float, float), the complete adjacency matrix of graph, adjacencyMatrix[i][j] contains the value of the arc connecting 
                        nodes i and j

@param : terminals : list() the list containing the index of terminal nodes

@param : populationSize : the number of individuals to generate

@param : n : the number of iterations
    
@param : M : Great number used to penalize bad solutions (the ones that don't return a tree but a forest)
    
@return : list() of shape : [individual : tuple(), fitness] : the best solution found with genetic algorithm 

"""  
def simpleGeneticAlgorithm(adjacencyMatrix, terminals, populationSize, n, M, f, j, population, startTime) :
    
#    if j<2 :
#        population = createRandomInitPopulation(len(adjacencyMatrix)-len(terminals), populationSize)
#    else :
#        population = createHeuristicInitPopulation(adjacencyMatrix, terminals, populationSize)
    fitnessList = []
#    launchTime = initTime
    lastTime = time.time()
    k=0
#    for k in range(n) :
    while time.time()-startTime < 240 :    
        print("********************************* k=",k)
        print("*************population size : ",len(population))
#        print(population)
        print("*******************categories : ", len(getCategories(population)))
        f.write("********************************* k="+str(k))
        f.write("*************population size : "+str(len(population)))
#        print(population)
        f.write("*******************categories : "+str(len(getCategories(population))))
        f.write("******************launch Time : "+str(int(time.time()-startTime)))
        fitnessList = []
        fitnessSum = 0
        
        
        # calculate and adapts fitness to fit the rule : the more the fit the more the individual's prob is
        for i in range(populationSize) :
            individual = tuple(population[i])
            fit = fitness(terminals, individual, adjacencyMatrix, M)
            fitnessList.append([individual, fit, 1/fit])#, 0, 0] # {individual : [fitness, new fitness, probability, cumprob]}
            fitnessSum += 1/fit

        newPopulation = elitisimBasedSurvivorSelection(fitnessList, fitnessSum, populationSize, j)
        print(">>>>>>>>>>>>>>>> new Individuals : ",getNbOfNewIndividuals(population, newPopulation))
        f.write("\n>>>>>>>>>>>>>>>> new Individuals : "+str(getNbOfNewIndividuals(population, newPopulation)))
        population = newPopulation
        bestIndividual = []

        for individual in fitnessList :
            if len(bestIndividual) == 0 or bestIndividual[1] > individual[1] :
                bestIndividual = [individual[0], individual[1]]
            i += 1
        print("******************************************* k ====",k," bestFit :", bestIndividual[1], 
              "\nlen of the solution : ", len(bestIndividual[0]), " and len of terminals : ",len(terminals))
        f.writelines("******************************************* k ===="+str(k)+" bestFit :"+str(bestIndividual[1])+
              "\nlen of the solution : "+str(len(bestIndividual[0]))+" and len of terminals : "+str(len(terminals)))
        launchTime = time.time() - startTime   
#        lastTime = time.time()
        f.write("\nLAUNCH TIME : "+str(launchTime)+" ms")
        print("LAUNCH TIME : "+str(launchTime)+" ms")
        k+=1
        
    bestIndividual = []
    for individual in fitnessList :
        if len(bestIndividual) == 0 or bestIndividual[1] > individual[1] :
            bestIndividual = [individual[0], individual[1]]
    
    return bestIndividual
        
M = 100000*100000
n = 200
populationSize = 100
#n = 100
#populationSize = 50


for i in range(1,19) :
    
    if i <10 :
        adjacencyMatrix, terminals = readInstance(os.getcwd()+"\instances\B\\b0"+str(i)+".stp")
    else :
        adjacencyMatrix, terminals = readInstance(os.getcwd()+"\instances\B\\b"+str(i)+".stp")
    for j in [0,2] :
        
        if j==0 :
            title = "_without heuristic"
            n = 200
            populationSize = 100
            initTime = time.time()
            population = createRandomInitPopulation(len(adjacencyMatrix)-len(terminals), populationSize)
            initTime = time.time()-initTime
           

        else :
            title = "_with heuristic"
            n = 100
            populationSize = 50
            startTime = time.time()
            population = createHeuristicInitPopulation(adjacencyMatrix, terminals, populationSize)
            initTime = time.time()-startTime
            

        for y in range(2) :    
            if y == 0 :
                title = title+"_without distinct"
            else :
                title = title+"_with distinct"
            f = open(os.getcwd()+"\instances\B\output complete with time\\b_"+str(i)+title+".txt", "w")
            f.write("------------------Population generation time : "+str(initTime)+"\n\n")
            startTime = time.time()
            bestIndividual = simpleGeneticAlgorithm(adjacencyMatrix, terminals, populationSize, n, M, f, j+y, population, startTime-initTime)            
            f.write("\n\n\n----------"+str(bestIndividual[0])+"   "+str(bestIndividual[1]))
            f.close()
    
    
    
#iterations = 50

#terminals = [20, 18, 24]
#individual = np.zeros((37))

#print(fitness(terminals, individual, adjacencyMatrix,M))
#print(terminals)
#print(adjacencyMatrix, terminals)





#print()
#print(fitness(terminals, np.array(([0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,1,0,1,0,0,1,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0])), adjacencyMatrix,M))


#Codage:  [1-0, 2-0, 3-0, 4-0, 5-0, 8-0, 10-0, 12-0, 13-0, 14-0, 15-0, 16-0, 17-0, 18-0, 20-1, 21-0, 22-0, 
#          24-1,1, 27-1, 29-1, 30-0, 31-0, 32-0, 33-0, 34-0, 35-1, 36-1, 37-1, 38-0, 42-0,  44-1,
#          45-1, 46-0, 47-1, 48-0, 50-0]
#         
#Liste de sommets :
# ['u22', 'u1', 'u15', 'u38',
#  'u45', 'u27', 'u18', 'u48', 'u36', 'u3', 'u8', 'u21', 'u17', 'u24', 'u50', 'u37', 'u33', 'u35', 'u5', 'u42', 'u20',
#  'u46', 'u26', 'u29', 'u14', 'u16', 'u30', 'u31', 'u47', 'u44', 'u10', 'u2', 'u13', 'u34', 'u4', 'u32', 'u12']

#parent1 = [0,1,0,0,0,0,0,0]
#parent2 = [1,0,1,1,1,1,1,1]
#
#print(onePoint_crossover(parent1, parent2))
#bitFlip_mutation(parent1)
#print(parent1)

matrix = np.zeros((5,5))
matrix[0] = [0,0,2,1,5]
matrix[1] = [0,0,6,4,2]
matrix[2] = [2,6,0,3,1]
matrix[3] = [1,4,3,0,2]
matrix[4] = [5,2,1,2,0]
#
#fitnessList = [[tuple([0,0,0]), 10],
#                [tuple([0,0,1]), 4],
#                [tuple([0,1,1]), 6],
#                [tuple([0,1,0]), 1],
#                [tuple([0,0,1]), 4]
#                ]
#
#print(bestDistinctIndividuals(fitnessList, 5))

#parent1 = [0,0,1,1,0]
#parent2 = [0,0,0,1,1]
#n=100

#print(createHeuristicInitPopulation(matrix, [0,1], 10))

#print(simpleGeneticAlgorithm(matrix, [0,1], n, M))

