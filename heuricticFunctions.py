'''
Created on 10. 4. 2018

@author: Magdalena
'''
from builtins import len

import numpy as np
from UnionFind import *
from readFile import *
import operator
from itertools import chain



def setOfArcsOnSubset(adjacencyMatrix, terminals):
    arcs = dict()
    
    for i in range(adjacencyMatrix.shape[0]):
        for j in range(i+1,adjacencyMatrix.shape[0]):
            if adjacencyMatrix[i][j] != 0 and adjacencyMatrix[i][j] != np.inf:
                arcs[(terminals[i],terminals[j])] = adjacencyMatrix[i][j]
                #arcs.append([i,j,adjacencyMatrix[i][j]])
    return arcs

def setOfArcs(adjacencyMatrix):
    arcs = dict()
    
    for i in range(adjacencyMatrix.shape[0]):
        for j in range(i+1,adjacencyMatrix.shape[0]):
            if adjacencyMatrix[i][j] != 0 and adjacencyMatrix[i][j] != np.inf:
                arcs[(i,j)] = adjacencyMatrix[i][j]
                #arcs.append([i,j,adjacencyMatrix[i][j]])
    return arcs

def KruskalOnSubset(adjacencyMatrix,terminals):
    UF = UnionFind(adjacencyMatrix)
    
    for i in range(adjacencyMatrix.shape[0]):
        UF.makeSet(terminals[i])

    arcs = setOfArcsOnSubset(adjacencyMatrix,terminals)
    print("arcs1: ",arcs)
    arcs = sorted(arcs.items(),key = operator.itemgetter(1))
    print("arcs: ", arcs)
    
    keys = [i[0] for i in arcs]
    values = [i[1] for i in arcs]
    spanningTree = dict()
    
    for i in range(len(arcs)):
        
        if UF.find(keys[i][0]) != UF.find(keys[i][1]):
            spanningTree[keys[i]] = values[i] 
            UF.union(keys[i][0], keys[i][1])
    
    return spanningTree

def Kruskal(adjacencyMatrix):
    UF = UnionFind(adjacencyMatrix)
    
    for i in range(adjacencyMatrix.shape[0]):
        UF.makeSet(i)

    arcs = setOfArcs(adjacencyMatrix)
    print("arcs1: ",arcs)
    arcs = sorted(arcs.items(),key = operator.itemgetter(1))
    print("arcs: ", arcs)
    
    keys = [i[0] for i in arcs]
    values = [i[1] for i in arcs]
    spanningTree = dict()
    
    for i in range(len(arcs)):
        
        if UF.find(keys[i][0]) != UF.find(keys[i][1]):
            spanningTree[keys[i]] = values[i] 
            UF.union(keys[i][0], keys[i][1])
    
    return spanningTree
    

def openedNodesArgMin(array, open_nodes):
    argmin = -1
    
    for i in range(len(array)):
        if open_nodes[i] and argmin == -1:
            argmin = i
            
        elif open_nodes[i] and array[i] < array[argmin] :
            argmin = i
    
    return argmin

def arrayMin(array):
    min = array[0]
    
    for i in range(len(array)):
        if(array[i] < array[min]):
            min = array[i]

    return min

def lengthOfShortestPath(startNode, finalNode, adjacenceMatrix):
    
    distances = [np.inf for i in range (adjacenceMatrix.shape[0])]
    distances[int(startNode)] = 0
    
    actNode = int(startNode)
    opened_nodes = [True for i in range(adjacenceMatrix.shape[0])]
    
    while (actNode != int(finalNode)) :
        opened_nodes[actNode] = False
        
        for j in range (len(adjacenceMatrix)) :
            if opened_nodes[j] and adjacenceMatrix[int(actNode)][j] != np.inf:
                distances[j] = min(distances[j], distances[int(actNode)]+adjacenceMatrix[int(actNode)][j])
                      
        actNode = openedNodesArgMin(distances, opened_nodes) 
    
    
    return distances[int(finalNode)]

def reconstructPath(startNode, finalNode, adjacencyMatrix, father):
    arcs = dict()
    
    actNode = int(finalNode)
    
    while(actNode != startNode):
        arcs[(actNode, int(father[actNode]))] = adjacencyMatrix[actNode][int(father[actNode])]
        actNode = int(father[actNode])
        
    return arcs

def Dijkstra(startNode, finalNode, adjacencyMatrix):
    
    distances = [np.inf for i in range (adjacencyMatrix.shape[0])]
    distances[int(startNode)] = 0
    
    actNode = int(startNode)
    opened_nodes = [True for i in range(adjacencyMatrix.shape[0])]
    father = [None for i in range (adjacencyMatrix.shape[0])]
        
    while (actNode != int(finalNode)) :
        opened_nodes[actNode] = False
        
        for j in range (len(adjacencyMatrix)) :
            if opened_nodes[j] and adjacencyMatrix[int(actNode)][j] != np.inf:
                if (distances[j] > distances[int(actNode)]+adjacencyMatrix[int(actNode)][j]):
                    distances[j] = distances[int(actNode)]+adjacencyMatrix[int(actNode)][j]
                    father[j] = actNode
                    
                      
        actNode = openedNodesArgMin(distances, opened_nodes) 
    
    path = reconstructPath(startNode, finalNode, adjacencyMatrix, father)
    
    
    return path, distances[int(finalNode)]

def distanceGraphHeuristic(adjacenceMatrix, terminals):
    distanceGraphAdjacenceMatrix = np.full((len(terminals), len(terminals)), np.inf)
    paths = [[None for j in range(len(terminals))] for i in range(len(terminals))]
    for i in range(len(terminals)):
        distanceGraphAdjacenceMatrix[i][i] = 0
    
    for i in range(len(terminals)):
        #print(i)
        for j in range(i+1,len(terminals)):
            #print("\t",j)
            paths[i][j], distanceGraphAdjacenceMatrix[i][j] = Dijkstra(terminals[i], terminals[j], adjacenceMatrix)
            distanceGraphAdjacenceMatrix[j][i] = distanceGraphAdjacenceMatrix[i][j]
            paths[j][i] = paths[i][j]
    
    print("------ PATHS: ",paths)
    
    g2 = KruskalOnSubset(distanceGraphAdjacenceMatrix, terminals)
    print("-------- G2 :", g2)
    
    g3 = g2.copy()
    arcs = g2.keys()
    
    for arc in g2:
        print("\t ------------- G3 before :",g3)
        g3.pop(arc)
        g3.update(paths[int(arc[0])][int(arc[1])])
        print("\t ------------- G3 after :",g3)
    
    g3Nodes = list(set(chain.from_iterable(g3.keys())))
    g3AdjacencyMatrix = np.full((len(g3Nodes), len(g3Nodes)), np.inf)
    
    for i in range(len(g3Nodes)):
        for j in range(len(g3Nodes)):
            g3AdjacencyMatrix[i][j] = adjacencyMatrix[g3Nodes[i]][g3Nodes[j]]
    
    g4 = KruskalOnSubset(g3AdjacencyMatrix, g3Nodes)
    
    print("------------------ G4: ",g4)
    
    degrees = [0 for i in range(adjacencyMatrix.shape[0])]
    
    arcs = g4.keys()
    
    for arc in arcs: 
        degrees[arc[0]] +=1
        degrees[arc[1]] +=1
        
    isTerminal = [False for i in range(adjacencyMatrix.shape[0])]
    for i in range(len(terminals)):
        isTerminal[int(terminals[i])] = True
        
    ntLeaves = nonTerminalLeaves(degrees, isTerminal)
    
    while(ntLeaves != []):
        newTree = g4.copy()
        print("spanningTree:", g4)
        print("degrees: ",degrees)
        print("ntLeaves: ", ntLeaves)
        for leaf in ntLeaves:
            for arc in g4:
                print("arc: ", arc)
                print("arc[0] = ",arc[0], " arc[1] = ",arc[1], "leaf = ",leaf)
                if arc[0] == leaf or arc[1]==leaf:
                    newTree.pop(arc,None)
                    degrees[arc[0]] -= 1
                    degrees[arc[1]] -= 1
                    print(newTree)
                    
        g4 = newTree
        
        ntLeaves = nonTerminalLeaves(degrees, isTerminal)
      
    return g4

def nonTerminalLeaves(degrees, isTerminal):
    res = []    
    for i in range(len(degrees)):
        if degrees[i] == 1 and not isTerminal[i]:  
            res.append(i)
    return res

def ACMHeuristic(adjacencyMatrix, terminals):
    isTerminal = [False for i in range(adjacencyMatrix.shape[0])]
    for i in range(len(terminals)):
        isTerminal[int(terminals[i])] = True
    
    spanningTree = Kruskal(adjacencyMatrix)
    nodesInSpanningTree = [True for i in range(adjacencyMatrix.shape[0])]
    degrees = [0 for i in range(len(nodesInSpanningTree))]
    
    arcs = spanningTree.keys()
    
    for arc in arcs: 
        degrees[arc[0]] +=1
        degrees[arc[1]] +=1
        
    ntLeaves = nonTerminalLeaves(degrees, isTerminal)
    
    
    
    while(ntLeaves != []):
        newTree = spanningTree.copy()
        print("spanningTree:", spanningTree)
        print("degrees: ",degrees)
        print("ntLeaves: ", ntLeaves)
        for leaf in ntLeaves:
            nodesInSpanningTree[leaf] = False
            for arc in spanningTree:
                print("arc: ", arc)
                print("arc[0] = ",arc[0], " arc[1] = ",arc[1], "leaf = ",leaf)
                if arc[0] == leaf or arc[1]==leaf:
                    newTree.pop(arc,None)
                    degrees[arc[0]] -= 1
                    degrees[arc[1]] -= 1
                    print(newTree)
                    
        spanningTree = newTree
        """
        arcs = spanningTree.keys()
        for i in range(len(degrees)):
            degrees[i] = 0
        
        for arc in arcs: 
            degrees[arc[0]] +=1
            degrees[arc[1]] +=1
        """
        ntLeaves = nonTerminalLeaves(degrees, isTerminal)
        
    return spanningTree    

"""
@param adjacencyMatrix: np.array((|V|,|V|)), the matrix of size |V|*|V|, the element [i][j] contains the cost of the arc between the nodes i and j
@param minBound: int, the lower bound for percentage of perturbation 
@param minBound: int, the upper bound for percentage of perturbation 

@return: newGraph:  np.array((|V|,|V|)), the matrix of size  |V|*|V|, the element [i][j] containts the perturbed cost of the arc between the nodes i and j 
"""
def inputGraphRandomization(adjacencyMatrix, minBound, maxBound):
    newGraph = np.zeros((adjacencyMatrix.shape[0],adjacencyMatrix.shape[1]))
    
    for i in range(adjacencyMatrix.shape[0]):
        for j in range(adjacencyMatrix.shape[1]):
            r = np.random.randint(minBound, maxBound)
            newGraph[i][j] = adjacencyMatrix[i][j] + adjacencyMatrix[i][j]*r*0.01
   
    return newGraph

""" TEST OF THE FUNCTION lengthOfShortestPath
matrix = [[0,85,217,np.inf, 173, np.inf, np.inf, np.inf, np.inf, np.inf],
          [85, 0, np.inf, np.inf, np.inf, 80, np.inf, np.inf, np.inf, np.inf],
          [217, np.inf, 0, np.inf, np.inf, np.inf, 186, 103, np.inf, np.inf],
          [np.inf, np.inf, np.inf, 0, np.inf, np.inf, np.inf, 183, np.inf, np.inf],
          [173, np.inf, np.inf, np.inf, 0, np.inf, np.inf, np.inf, np.inf, 502],
          [np.inf, 80, np.inf, np.inf, np.inf, 0, np.inf, np.inf, 250, np.inf],
          [np.inf, np.inf, 186, np.inf, np.inf, np.inf, 0, np.inf, np.inf, np.inf],
          [np.inf, np.inf, 103, 183, np.inf, np.inf, np.inf, 0, np.inf, 167],
          [np.inf, np.inf, np.inf, np.inf, np.inf, 250, np.inf, np.inf, 0, np.inf],
          [np.inf, np.inf, np.inf, np.inf, 502, np.inf, np.inf, 167, 84, 0]]

print(lengthOfShortestPath(0, 8, matrix))

"""

adjacencyMatrix, terminals = readInstance(os.getcwd()+"\heuristic\instance_test.gr")
print(adjacencyMatrix, terminals)
print(lengthOfShortestPath(0, 1, adjacencyMatrix))
dgMatrix = distanceGraphHeuristic(adjacencyMatrix, terminals)


ACMHeuristic(adjacencyMatrix, terminals)
inputGraphRandomization(adjacencyMatrix, 5, 20)
