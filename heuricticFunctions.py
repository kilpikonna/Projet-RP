'''
Created on 10. 4. 2018

@author: Magdalena
'''
 # coding=utf8

from builtins import len

import numpy as np
from UnionFind import *
from readFile import *
import operator
from itertools import chain


"""
@param : nodesSubset : list, the subset of all graph nodes ordered by nodes id - nodesSubset[i] containts the id of the (i+1)-th "smallest" node 
                        in the subset 
                        F.E. : Considernig the graph with 10 nodes 0, 1, ..., 9
                               with the subset of nodes {2,5,7,8}, we will have:
                               nodesSubset[0] = 2
                               nodesSubset[1] = 5
                               nodesSubset[2] = 7
                               nodesSubset[3] = 8
@param : adjacencyMatrix : np.array(float, float) - the sub-matrix of the graph complete adjacency matrix; containts only the lines and columns of the nodes
                           selected in the nodesSubset.
                           With the previous example, the complete graph adjacency matrix will have 10 lines and columns
                           The parametrs adjacencyMatrix will be its submatrix of the shape 4*4 containing the intersection of its 
                           3rd, 5th, 8th and 9th lines and columns
@return : arcs, dict - keys are the couples of arcs connecting two nodes of nodesSubset, values are the values of these arcs.
          For example, if the graph from the previous example contains the arc (2,5) of the value 1, we will have:
          arcs[(2,5)] == 1
"""
def setOfArcsOnSubset(adjacencyMatrix, nodesSubset):
    arcs = dict()
    
    for i in range(len(nodesSubset)):
        for j in range(i+1,adjacencyMatrix.shape[0]):
            if adjacencyMatrix[i][j] != 0 and adjacencyMatrix[i][j] != np.inf:
                arcs[(nodesSubset[i],nodesSubset[j])] = adjacencyMatrix[i][j]
                #arcs.append([i,j,adjacencyMatrix[i][j]])
    return arcs

"""
@param adjacencyMatrix: np.array(float, float), the complete adjacency matrix of graph, adjacencyMatrix[i][j] contains the value of the arc connecting 
                        nodes i and j
                        we have:    adajacencyMatrix[i][i] = 0
                                    adjacencyMatrix[i][j] = + inf if nodes i and j aren't connected
@return arcs: dict(), the graph represented by a set of its arcs - for each couple of nodes i and j that are connected by an arc,
                      the dict associates the value adjacencyMatrix[i][j] to the key (i,j).
"""

def setOfArcs(adjacencyMatrix):
    arcs = dict()
    
    for i in range(adjacencyMatrix.shape[0]):
        for j in range(i+1,adjacencyMatrix.shape[0]):
            if adjacencyMatrix[i][j] != 0 and adjacencyMatrix[i][j] != np.inf:
                arcs[(i,j)] = adjacencyMatrix[i][j]
                #arcs.append([i,j,adjacencyMatrix[i][j]])
    return arcs

"""
@param : nodesSubset : list, the subset of all graph nodes ordered by nodes id - nodesSubset[i] containts the id of the (i+1)-th "smallest" node 
                        in the subset 
                        F.E. : Considernig the graph with 10 nodes 0, 1, ..., 9
                               with the subset of nodes {2,5,7,8}, we will have:
                               nodesSubset[0] = 2
                               nodesSubset[1] = 5
                               nodesSubset[2] = 7
                               nodesSubset[3] = 8
@param : adjacencyMatrix : np.array(float, float) - the sub-matrix of the graph complete adjacency matrix; containts only the lines and columns of the nodes
                           selected in the nodesSubset.
                           With the previous example, the complete graph adjacency matrix will have 10 lines and columns
                           The parametrs adjacencyMatrix will be its submatrix of the shape 4*4 containing the intersection of its 
                           3rd, 5th, 8th and 9th lines and columns
                           
@return : spanningTree : dict() - the spanning tree of the subgraph represented as a set of arcs, using the dict() structure
          This spanning tree is found using the Kruskal algorithm, for the efficiency reason Kruskal algorithm uses 
          the Union-Find structure
"""
def KruskalOnSubset(adjacencyMatrix, nodesSubset):
    UF = UnionFind(adjacencyMatrix)
    
    for i in range(len(nodesSubset)):
        UF.makeSet(nodesSubset[i])

    arcs = setOfArcsOnSubset(adjacencyMatrix,nodesSubset)
    
    arcs = sorted(arcs.items(),key = operator.itemgetter(1))
        
    keys = [i[0] for i in arcs]
    values = [i[1] for i in arcs]
    spanningTree = dict()
    
    for i in range(len(arcs)):
        
        if UF.find(keys[i][0]) != UF.find(keys[i][1]):
            spanningTree[keys[i]] = values[i] 
            UF.union(keys[i][0], keys[i][1])
    
    return spanningTree
"""
@param adjacencyMatrix: np.array(float, float), the complete adjacency matrix of graph, adjacencyMatrix[i][j] contains the value of the arc connecting 
                        nodes i and j
                        we have:    adajacencyMatrix[i][i] = 0
                                    adjacencyMatrix[i][j] = + inf if nodes i and j aren't connected

@return : spanningTree : dict() - the spanning tree of the graph represented as a set of arcs, using the dict() structure
          This spanning tree is found using the Kruskal algorithm, for the efficiency reason Kruskal algorithm uses 
          the Union-Find structure                                    
"""
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
    
"""
@summary: This is an auxiliary function for the Dijkstra algorithm which returns the argmin of array. It is called during the
          Dijkstra algorithm execution

@param nodesEvaluation : list, nodesEvaluation[i] contains the evaluation by the Dijkstra algorithm of the node i
@param openedNodes :     list, openedNodes[i] =     True     if the node i is opened
                                                    False    if the node i is closed

@return argmin : the index of the node with the lowest evaluation 
"""
def openedNodesArgMin(nodesEvaluation, openedNodes):
    argmin = -1
    
    for i in range(len(nodesEvaluation)):
        if openedNodes[i] and argmin == -1:
            argmin = i
            
        elif openedNodes[i] and nodesEvaluation[i] < nodesEvaluation[argmin] :
            argmin = i
    
    return argmin

"""
@summary: This is an auxiliary function for the Dijkstra algorithm which returns the argmin of array. It is called during the
          Dijkstra algorithm execution

@param array: list(float) - the list of numbers (it this context the node evaluations)
@return: min : the element minimum of the array
"""
def arrayMin(array):
    min = array[0]
    
    for i in range(len(array)):
        if(array[i] < array[min]):
            min = array[i]

    return min

#def lengthOfShortestPath(startNode, finalNode, adjacenceMatrix):
#    
#    distances = [np.inf for i in range (adjacenceMatrix.shape[0])]
#    distances[int(startNode)] = 0
#    
#    actNode = int(startNode)
#    opened_nodes = [True for i in range(adjacenceMatrix.shape[0])]
#    
#    while (actNode != int(finalNode)) :
#        opened_nodes[actNode] = False
#        
#        for j in range (len(adjacenceMatrix)) :
#            if opened_nodes[j] and adjacenceMatrix[int(actNode)][j] != np.inf:
#                distances[j] = min(distances[j], distances[int(actNode)]+adjacenceMatrix[int(actNode)][j])
#                      
#        actNode = openedNodesArgMin(distances, opened_nodes) 
#    
#    
#    return distances[int(finalNode)]

"""
@param startNode: int, the node of departure
@param finalNode: int, the node we want to reach
@param adjacencyMatrix: np.array(float, float), the complete adjacency matrix of graph, adjacencyMatrix[i][j] contains the value of the arc connecting 
                        nodes i and j
                        we have:    adajacencyMatrix[i][i] = 0
                                    adjacencyMatrix[i][j] = + inf if nodes i and j aren't connected
@param father: list(int), father[i] containts the predecessor of the node i on the path - enable the path reconstruction

@return: arcs : dict() : the shortest path represented as a set of arcs. If the arc (i,j) of the value v is on the path,
                         we have arcs[(i,j)] == v
"""
def reconstructPath(startNode, finalNode, adjacencyMatrix, father):
    arcs = dict()
    
    actNode = int(finalNode)
    
    while(actNode != startNode):
        arcs[(actNode, int(father[actNode]))] = adjacencyMatrix[actNode][int(father[actNode])]
        actNode = int(father[actNode])
        
    return arcs

"""
@param startNode: int, the node of departure
@param finalNode: int, the node we want to reach
@param adjacencyMatrix: np.array(float, float), the complete adjacency matrix of graph, adjacencyMatrix[i][j] contains the value of the arc connecting 
                        nodes i and j
                        we have:    adajacencyMatrix[i][i] = 0
                                    adjacencyMatrix[i][j] = + inf if nodes i and j aren't connected
@return (path, distances[finalNode]) : (dict, float) : the shortest path connecting the startNode and finalNode (= the set of arcs
                                        represented by a dict() ), the length of this path
"""
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

"""
@summary:The heuristic described in the section 2.1

@param adjacenceMatrix: np.array(float, float), the complete adjacency matrix of graph, adjacencyMatrix[i][j] contains the value of the arc connecting 
                        nodes i and j
                        we have:    adajacencyMatrix[i][i] = 0
                                    adjacencyMatrix[i][j] = + inf if nodes i and j aren't connected
@param terminals: list() the list containing the index of terminal nodes
@return: spanningTree : dict() - the spanning tree of the subgraph represented as a set of arcs, using the dict() structure
         It's a solution of the Steiner problem
"""
def distanceGraphHeuristic(adjacencyMatrix, terminals):
    distanceGraphAdjacenceMatrix = np.full((len(terminals), len(terminals)), np.inf)
    paths = [[None for j in range(len(terminals))] for i in range(len(terminals))]
    for i in range(len(terminals)):
        distanceGraphAdjacenceMatrix[i][i] = 0
    
    for i in range(len(terminals)):
        #print(i)
        for j in range(i+1,len(terminals)):
            #print("\t",j)
            paths[i][j], distanceGraphAdjacenceMatrix[i][j] = Dijkstra(terminals[i], terminals[j], adjacencyMatrix)
            distanceGraphAdjacenceMatrix[j][i] = distanceGraphAdjacenceMatrix[i][j]
            paths[j][i] = paths[i][j]
    
    #print("------ PATHS: ",paths)
    
    #g2 = KruskalOnSubset(adjacencyMatrix, terminals)
    g2 = Kruskal(distanceGraphAdjacenceMatrix)
    #print("-------- G2 :", g2)
    
    g3 = g2.copy()
    arcs = g2.keys()
    
    for arc in g2:
        #print("\t ------------- G3 before :",g3)
        g3.pop(arc)
        g3.update(paths[int(arc[0])][int(arc[1])])
        #print("\t ------------- G3 after :",g3)
    print("G3 nodes: ", g3)
    g3Nodes = list(set(chain.from_iterable(g3.keys())))
    g3AdjacencyMatrix = np.full((len(g3Nodes), len(g3Nodes)), np.inf)
    
    for i in range(len(g3Nodes)):
        for j in range(len(g3Nodes)):
            g3AdjacencyMatrix[i][j] = adjacencyMatrix[g3Nodes[i]][g3Nodes[j]]
    
    #g4 = Kruskal(adjacencyMatrix, g3Nodes)
    #g4 = KruskalOnSubset(g3AdjacencyMatrix, g3Nodes)
    g4 = Kruskal(g3AdjacencyMatrix)
    #print("------------------ G4: ",g4)
    
    degrees = [0 for i in range(adjacencyMatrix.shape[0])]
    
    arcs = g4.keys()
    
    for arc in arcs: 
        degrees[g3Nodes[arc[0]]] +=1
        degrees[g3Nodes[arc[1]]] +=1
        
    isTerminal = [False for i in range(adjacencyMatrix.shape[0])]
    for i in range(len(terminals)):
        isTerminal[int(terminals[i])] = True
        
    ntLeaves = nonTerminalLeaves(degrees, isTerminal)
    
    while(ntLeaves != []):
        newTree = g4.copy()
        #print("spanningTree:", g4)
        print("degrees: ",degrees)
        print("ntLeaves: ", ntLeaves)
        for leaf in ntLeaves:
            for arc in g4:
                #print("arc: ", arc)
                #print("arc[0] = ",arc[0], " arc[1] = ",arc[1], "leaf = ",leaf)
                if g3Nodes[arc[0]] == leaf or g3Nodes[arc[1]]==leaf:
                    newTree.pop(arc,None)
                    degrees[g3Nodes[arc[0]]] -= 1
                    degrees[g3Nodes[arc[1]]] -= 1
                    print(newTree)
                    
        g4 = newTree
        
        ntLeaves = nonTerminalLeaves(degrees, isTerminal)
      
    arcs = g4.keys()
    #print(g4)
    g5 = dict()
    g5
    for arc in arcs:
        g5[(g3Nodes[arc[0]], g3Nodes[arc[1]])] = g4[arc]
        
    #print(g5)
    return g5

"""
@param degrees: list(int), degrees[i] contains the degree of the node i in the constructed spanning tree
@param isTerminal: isTerminal[i] contains True if i is a terminal node, False otherwise
@return: res : list(int), the list of non-terminal leaves of the degree 1
"""
def nonTerminalLeaves(degrees, isTerminal):
    res = []    
    for i in range(len(degrees)):
        if degrees[i] == 1 and not isTerminal[i]:  
            res.append(i)
    return res

"""
@summary:The heuristic described in the section 2.2

@param adjacenceMatrix: np.array(float, float), the complete adjacency matrix of graph, adjacencyMatrix[i][j] contains the value of the arc connecting 
                        nodes i and j
                        we have:    adajacencyMatrix[i][i] = 0
                                    adjacencyMatrix[i][j] = + inf if nodes i and j aren't connected
@param terminals: list() the list containing the index of terminal nodes
@return: spanningTree : dict() - the spanning tree of the subgraph represented as a set of arcs, using the dict() structure
         It's a solution of the Steiner problem
"""
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

#adjacencyMatrix, terminals = readInstance(os.getcwd()+"\heuristic\instance_test.gr")
#print(adjacencyMatrix, terminals)
##print(lengthOfShortestPath(0, 1, adjacencyMatrix))
#dgMatrix = distanceGraphHeuristic(adjacencyMatrix, terminals)
#
#
#ACMHeuristic(adjacencyMatrix, terminals)
#inputGraphRandomization(adjacencyMatrix, 5, 20)
