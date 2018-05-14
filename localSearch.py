'''
Created on 17. 4. 2018

@author: Magdalena
'''
from readFile import *

import numpy as np
from scipy.constants.constants import degree
from heuricticFunctions import *
from future.types.newmemoryview import newmemoryview
import time
import math
import matplotlib.pyplot as plt

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
            print("SOLUTION: ", solutionDict)
            
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


def GetArticulationPoints(startNode, d,visited, depth, low, parent, AM, res):
    visited[startNode] = True
    depth[startNode] = d
    low[startNode] = d
    
    childCount = 0
    isArticulation = False
    
    for i in range(0,AM.shape[0]):
        #print("Adjacency matrix shape: ", AM.shape)
        #print("startNode = ", startNode)
        #print("i = ", i)
        if AM[i][startNode] != np.inf and i != startNode:
            if (not visited[i]):
                #print("\t i: ", i, " start node: ", startNode)
                parent[i] = startNode
                #print("\t \t parent: ",parent)
                #print("low before:",low)
                #print("---------------------- NEW APPEL ------------------------")
                res += GetArticulationPoints(i, d+1, visited, depth, low, parent, AM, res)
                #print("--------------------------- RETURN ------------------------")
                #print("low after: ", low)  
                #print(res)
                res = list(set(res))
                childCount +=1
            
                if low[i] >= depth[startNode]:
                    isArticulation = True
                    #print("i is articulation! ",startNode)
            
                if low[startNode] > low[i]:
                    low[startNode] = low[i]
                
            elif i != parent[startNode] and depth[i] < low[startNode]:
                low[startNode] = depth[i]
                                
    if (parent[startNode] != None and isArticulation) or (parent[startNode] == None and childCount > 1):
        res += [startNode]
    
    #print("start node= ", startNode," parent = ", parent, " d = ",depth, " l = ",low)
    return res


def subMatrix(adjacencyMatrix, indexes):
    
    newMatrix = np.full((len(indexes), len(indexes)), np.inf)
    
    for i in range(newMatrix.shape[0]):
        for j in range(newMatrix.shape[1]):
            newMatrix[i][j] = adjacencyMatrix[indexes[i]][indexes[j]]
    
    return newMatrix
       
def createNeighbourhood(adjacencyMatrix, solution):
    degrees = calculateDegrees(adjacencyMatrix, solution)
    #print("Degrees: ", degrees)
    #print("Adjacency matrix :",adjacencyMatrix)
    
    neighbourhood = []
    indexesInSolution = [i for i in range(len(solution)) if solution[i]]
    visited = [False for i in range(len(indexesInSolution))]
    depth = [len(indexesInSolution) for i in range(len(indexesInSolution))]
    low = [len(indexesInSolution) for i in range(len(indexesInSolution))]
    parent = [None for i in range(len(indexesInSolution))]
    APtemp = GetArticulationPoints(0, 0, visited, depth, low, parent, subMatrix(adjacencyMatrix, indexesInSolution), [])
    
    AP = []
    
    for i in range(len(APtemp)):
        AP += [indexesInSolution[APtemp[i]]]
    
    for i in range(len(nonTerminalCoding)):
        #Elimination
        if solution[nonTerminalCoding[i]]:
            #We verify if the graph rests connected after the elimination of this node
            if nonTerminalCoding[i] not in AP:
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
    
    #print(neighbourhood)
       
    nodesSubset = [i for i in range(len(solution)) if solution[i]]
    arcs = KruskalOnSubset(adjacencyMatrix, nodesSubset)
    
    cost = 0
    values = arcs.values()
    
    for value in values:
        cost += value
    print(cost)
    best_found = solution
    best_cost = cost
    
    iteration = 0
    iterations_list = []
    changed = True
    startTime = time.time()
    
    while(changed and time.time() - startTime < 300):
        changed = False
        iteration += 1
        
        neighbourhood = createNeighbourhood(adjacencyMatrix, best_found)
        for sol in neighbourhood:
            nodesSubset = [i for i in range(len(sol)) if sol[i]]
            arcs = KruskalOnSubset(adjacencyMatrix, nodesSubset)
            #print("arcs donnes par Kruskal: ", arcs)
            cost = 0
            values = arcs.values()
    
            for value in values:
                cost += value
            #print(cost)
        
            if cost < best_cost:
                best_found = sol
                best_cost = cost
                changed = True
        print("Iteration: ", iteration)
        print(best_cost)
        #print(best_found)
        #print(terminals)
        iterations_list+=[best_cost]
        print([i for i in range(len(best_found)) if best_found[i]])
    #solutionCost(adjacencyMatrix,solution)    
    return best_cost, iterations_list

def localSearchRandom(adjacencyMatrix, terminals, solution):
    
    #print(neighbourhood)
       
    nodesSubset = [i for i in range(len(solution)) if solution[i]]
    arcs = KruskalOnSubset(adjacencyMatrix, nodesSubset)
    
    cost = 0
    values = arcs.values()
    
    for value in values:
        cost += value
    print(cost)
    best_found = solution
    best_cost = cost
    
    iteration = 0
    iterations_list = []
    changed = True
    startTime = time.time()
    
    while(changed and (time.time() - startTime < 300)):
        changed = False
        iteration += 1
        
        neighbourhood = createNeighbourhood(adjacencyMatrix, best_found)
        random_neigh = neighbourhood[np.random.randint(0, len(neighbourhood))]
        
        neighbourhood += createNeighbourhood(adjacencyMatrix, random_neigh)
        
        for sol in neighbourhood:
            nodesSubset = [i for i in range(len(sol)) if sol[i]]
            arcs = KruskalOnSubset(adjacencyMatrix, nodesSubset)
            #print("arcs donnes par Kruskal: ", arcs)
            cost = 0
            values = arcs.values()
    
            for value in values:
                cost += value
            #print(cost)
        
            if cost < best_cost:
                best_found = sol
                best_cost = cost
                changed = True
        print("Iteration: ", iteration)
        print(best_cost)
        #print(best_found)
        #print(terminals)
        iterations_list+=[best_cost]
        print([i for i in range(len(best_found)) if best_found[i]])
    #solutionCost(adjacencyMatrix,solution)    
    return best_cost, iterations_list
    
def localSearchLargeNeighbourhood(adjacencyMatrix, terminals, solution):
    
    #print(neighbourhood)
       
    nodesSubset = [i for i in range(len(solution)) if solution[i]]
    arcs = KruskalOnSubset(adjacencyMatrix, nodesSubset)
    iterations = []
    cost = 0
    values = arcs.values()
    
    for value in values:
        cost += value
    #print(cost)
    best_found = solution
    best_cost = cost
    
    iteration = 0
    changed = True
    startTime = time.time()
    
    while(changed and (time.time() - startTime < 300)):
        changed = False
        iteration += 1
        
        simple_neighbourhood = createNeighbourhood(adjacencyMatrix, best_found)
        neighbourhood = simple_neighbourhood.copy()
        
        for sol in simple_neighbourhood:
            #print("len neigh: ",len(neighbourhood))
            neighbourhood += createNeighbourhood(adjacencyMatrix, sol)
            
            
        for sol in neighbourhood:
            nodesSubset = [i for i in range(len(sol)) if sol[i]]
            arcs = KruskalOnSubset(adjacencyMatrix, nodesSubset)
    
            cost = 0
            values = arcs.values()
    
            for value in values:
                cost += value
            #print(cost)
        
            if cost < best_cost:
                best_found = sol
                best_cost = cost
                changed = True
        print("Iteration: ", iteration)
        print(best_cost)
        #print(best_found)
        #print(terminals)
        iterations += [best_cost]
        print([i for i in range(len(best_found)) if best_found[i]])
    #solutionCost(adjacencyMatrix,solution)  
    return best_cost, iterations  

def recuit_simule(adjacencyMatrix, terminals, solution):
    T0 = 10
    T_fin = 0.0001
    nb_it = 1
    nb_max = 1000
    tau = 1000000
    T = T0*math.exp(-nb_it/tau)
    print(math.exp(-1/tau))
    iteratoins = []
    
    nodesSubset = [i for i in range(len(solution)) if solution[i]]
    arcs = KruskalOnSubset(adjacencyMatrix, nodesSubset)
    
    cost = 0
    values = arcs.values()
    
    for value in values:
        cost += value
    #print(cost)
    best_found = solution
    best_cost = cost
    print(T, T_fin)
    startTime = time.time()
    
    while T > T_fin and nb_it < nb_max and time.time() - startTime < 300:
        #print(T, nb_it)
        print(best_cost)
        nb_it += 1   
        simple_neighbourhood = createNeighbourhood(adjacencyMatrix, best_found)
        new_solution = simple_neighbourhood[np.random.randint(0, len(simple_neighbourhood))]
        
        nodesSubset = [i for i in range(len(new_solution)) if new_solution[i]]
        arcs = KruskalOnSubset(adjacencyMatrix, nodesSubset)
    
        cost = 0
        values = arcs.values()
    
        for value in values:
            cost += value
            #print(cost)
        
        
        if cost < best_cost:
            best_cost = cost
            best_found = new_solution
        else:
            delta = cost - best_cost
            p = math.exp(-delta/T)
            
            if p > np.random.uniform(0,1):
                best_cost = cost
                best_found = new_solution
        
        T = T0*math.exp(-nb_it/tau)
        iteratoins += [best_cost]
    return best_cost, iteratoins

            
        
def test(instances_folder, output_file, localSearchHeuristic, initHeuristic="ACM", randomized = False, minBound = 5, maxBound = 20):
    #find files
    of = open(output_file, "w")
    nb_inst = 0
    for file in os.listdir(instances_folder):
        if file.endswith(".stp"):
            nb_inst += 0
            print("NEW INSTANCE :"+str(nb_inst))
            of.write("--------------------------------\n")
            of.write("Instance: "+file+"\n")
            adjacencyMatrix, terminals = readInstance(os.path.join(instances_folder, file))
            startTime = time.time()
            solution = initialSolution(adjacencyMatrix, terminals, initHeuristic, randomized, minBound, maxBound)
            
            if(localSearchHeuristic == "localSearchSimple"):
                best_found, it = localSearchSimple(adjacencyMatrix, terminals, solution)
                stopTime = time.time()
                sol = "solution: "+str(best_found)
                of.write(sol+"\n")
                tm = "time "+str(stopTime - startTime)
                of.write(tm+"\n")
                for i in range(len(it)):
                    of.write("\t iteration "+str(i+1)+" :"+str(it[i])+"\n")
                
            if(localSearchHeuristic == "localSearchLargeNeighbourhood"):
                best_found, it = localSearchLargeNeighbourhood(adjacencyMatrix, terminals, solution)
                stopTime = time.time()
                sol = "solution: "+str(best_found)
                of.write(sol+"\n")
                tm = "time "+str(stopTime - startTime)
                of.write(tm+"\n")
                for i in range(len(it)):
                    of.write("\t iteration :"+str(i+1)+" : "+str(it[i])+"\n")
                    
            if(localSearchHeuristic == "RS"):
                best_found, it = recuit_simule(adjacencyMatrix, terminals, solution)
                stopTime = time.time()
                sol = "solution: "+str(best_found)
                of.write(sol+"\n")
                tm = "time "+str(stopTime - startTime)
                of.write(tm+"\n")
                #for i in range(len(it)):
                #    of.write("\t iteration :"+str(i+1)+" : "+str(it[i])+"\n")
                
            if(localSearchHeuristic == "localSearchRandom"):
                best_found, it = localSearchRandom(adjacencyMatrix, terminals, solution)
                stopTime = time.time()
                sol = "solution: "+str(best_found)
                of.write(sol+"\n")
                tm = "time "+str(stopTime - startTime)
                of.write(tm+"\n")
                for i in range(len(it)):
                    of.write("\t iteration "+str(i+1)+" :"+str(it[i])+"\n")
    
    of.close()    
"""    
adjacencyMatrix = np.zeros((10,10))
terminals = [1, 5, 7]

initialSolution(adjacencyMatrix, terminals)
print(nonTerminalCoding)
"""

def graphiques(file, title, xTitle, yTitle):
    f = open(file)
    
    values_it = [0 for i in range(100)]
    
    opts = [85, 144, 754, 1079, 1579, 55, 102, 509, 707, 1093, 32, 46, 258, 323, 556, 11, 18, 113, 146, 267]
    sol = -1
    cpt = [0 for i in range(100)]
    line = f.readline()
    while line.strip() != "END":
        if line.split(":")[0] == "solution":
           line = f.readline()
           line = f.readline()
           sol += 1
           i = 0
           while not line.startswith("-"):
               print(line)
               values_it[i] += float(line.split(":")[1].strip())/float(opts[sol])
               cpt[i] += 1
               print(values_it)
               i +=1
               line = f.readline()
        else:
            line = f.readline()
            print(line)
    
    print("dehors")
    print(values_it)
    print(len(values_it))
    for i in range(len(values_it)):
        if(cpt[i] != 0):
            values_it[i] /= cpt[i]
    
    
    new = []
    for i in range(len(values_it)):
        if values_it[i] != 0:
            new.append(values_it[i])
    
    plt.plot([i+1 for i in range(len(new))], new, '-o', color="red",)
    plt.xlabel("numero d'iteration")
    plt.ylabel("ecart de l'optimum")
    plt.show()
    print(values_it)

#adjacencyMatrix, terminals = readInstance(os.getcwd()+"\instances\B\\b09.stp")
#solution = initialSolution(adjacencyMatrix, terminals, "PCC")
#print("Solution de depart :", solution)

#localSearchSimple(adjacencyMatrix, terminals, solution)
#print("---------------------------- LARGE NEIGHBOURHOOD ----------------------------")
#localSearchLargeNeighbourhood(adjacencyMatrix, terminals, solution)

#test(os.getcwd()+"\instances\\C", "C_rs_ACM.txt", "RS", "ACM")
#graphiques(os.getcwd()+"\C_random_ACM.txt", "", "", "")
