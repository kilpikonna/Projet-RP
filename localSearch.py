'''
Created on 17. 4. 2018

@author: Magdalena
'''
from readFile import *

import numpy as np
from scipy.constants.constants import degree
from heuricticFunctions import *

nonTerminalCoding = []

def initialSolution(adjacencyMatrix, terminals, heuristic, randomized = False, minBound = 5, maxBound = 20):
    global nonTerminalCoding
    nonTerminalCoding = [i for i in range(adjacencyMatrix.shape[0])]
    
    for i in range(len(terminals)):
        nonTerminalCoding.remove(terminals[i])
        
    if heuristic == "PCC":
        if randomized:
            randomizedGraph = inputGraphRandomization(adjacencyMatrix, minBound,maxBound)
            solutionDict = distanceGraphHeuristic(randomizedGraph, terminals)
        else:
            solutionDict = distanceGraphHeuristic(adjacencyMatrix, terminals)
            
    elif heuristic == "ACM":
        if randomized:
            randomizedGraph = inputGraphRandomization(adjacencyMatrix, minBound,maxBound)
            solutionDict = ACMHeuristic(randomizedGraph, terminals)
        else:
            solutionDict = ACMHeuristic(adjacencyMatrix, terminals)
    else:
        if randomized:
            randomizedGraph = inputGraphRandomization(adjacencyMatrix, minBound,maxBound)
            solutionDict = Kruskal(randomizedGraph)
        else:
            solutionDict = Kruskal(adjacencyMatrix)
    
    arcs = solutionDict.keys()
    
    solution = [False for i in range(adjacencyMatrix.shape[0])]
    
    for arc in arcs:
        if not solution[arc[0]]:
            solution[arc[0]] = True
            
        if not solution[arc[1]]:    
            solution[arc[1]] = True
        
    return solution

"""
@param solution: list(boolean) - solution[i] =  True if the node i belongs to the set S u T
                                                False otherwise
"""
#def randomNeighbour(solution):
#    i = np.random.randint(0, len(nonTerminalCoding))

def calculateDegrees(adjacencyMatrix, solution): 
    degrees = [0 for i in range(adjacencyMatrix.shape[0])]
    
    #We calculate the degree of each node in S u T 
    for i in range(adjacencyMatrix.shape[0]):
        for j in range(adjacencyMatrix.shape[0]):
            if i != j and solution[i] and solution[j] and adjacencyMatrix[i][j] != np.inf:
                degrees[i] += 1 
    
    return degrees

"""
@param solution: list(boolean) - solution[i] =  True if the node i belongs to the set S u T
                                                False otherwise
"""    
def restsConnected(adjacencyMatrix, degrees, solution, nodeToEliminate):    
    #For each node:
    for i in range(len(degrees)):
        #If the node is of degree 1 and the only node it is connected to is nodeToEliminate, the graph does not rest connected 
        #after elimination
        if i != nodeToEliminate and solution[i] and degrees[i] == 1 and adjacencyMatrix[i][nodeToEliminate] != np.inf:
            return False
     
    return True
       
def createNeighbourhood(adjacencyMatrix, solution):
    degrees = calculateDegrees(adjacencyMatrix, solution)
    print("Degrees: ", degrees)
    print("Adjacency matrix :",adjacencyMatrix)
    
    neighbourhood = []
    for i in range(len(nonTerminalCoding)):
        #Elimination
        if solution[nonTerminalCoding[i]]:
            #We verify if the graph rests connected after the elimination of this node
            if restsConnected(adjacencyMatrix, degrees, solution, nonTerminalCoding[i]):
                newSolution = solution.copy()
                newSolution[nonTerminalCoding[i]] = False
                neighbourhood.append(newSolution)
        
        #Insertion
        if not solution[nonTerminalCoding[i]]:
            deg = 0
            for j in range(adjacencyMatrix.shape[1]):
                if solution[j] and j != nonTerminalCoding[i] and adjacencyMatrix[nonTerminalCoding[i]][j] != np.inf:
                    deg += 1
            
            if deg > 1:
                newSolution = solution.copy()
                newSolution[nonTerminalCoding[i]] = True
                neighbourhood.append(newSolution)
    
    return neighbourhood
    
def localSearchSimple(adjacencyMatrix, terminals, solution):
    neighbourhood = createNeighbourhood(adjacencyMatrix, solution)
    print(neighbourhood)
        
        
"""    
adjacencyMatrix = np.zeros((10,10))
terminals = [1, 5, 7]

initialSolution(adjacencyMatrix, terminals)
print(nonTerminalCoding)
"""

adjacencyMatrix, terminals = readInstance(os.getcwd()+"\heuristic\instance_test.gr")
solution = initialSolution(adjacencyMatrix, terminals, "ACM")
print("Solution de depart :", solution)

localSearchSimple(adjacencyMatrix, terminals, solution)