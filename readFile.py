import os
import numpy as np


def readEdges(file, nbNodes, nbEdges):
    
    adjacencyMatrix = np.zeros((nbNodes,nbNodes)).astype(int)
    
    for i in range(nbEdges):
        lineSplitted = file.readline().strip().split(" ")
        
        if(lineSplitted[0] != "E"):
            print("******************* ERROR FILE FORMAT ************************* ")
            return None
        
        #else
        adjacencyMatrix[int(lineSplitted[1])-1][int(lineSplitted[2])-1] = int(lineSplitted[3])
        #graph is not oriented
        adjacencyMatrix[int(lineSplitted[2])-1][int(lineSplitted[1])-1] = int(lineSplitted[3])

    return adjacencyMatrix, file

def readTerminals(file, nbTerminals):
    
    terminalsList = np.zeros(nbTerminals)
    
    for i in range(nbTerminals):
        lineSplitted = file.readline().strip().split(" ")
        
        if(lineSplitted[0] != "T"):
            print("******************* ERROR FILE FORMAT ************************* ")
            return None
        #else
        terminalsList[i] = int(lineSplitted[1])
    
    return terminalsList, file

def readInstance(fileName):
    file = open(fileName, "r")
    line = file.readline() 

    nbNodes = 0
    nbEdges = 0
    nbTerminals = 0
     
    while line.strip() != "EOF":
        
        if "SECTION Graph" in line:
            nbNodes = int(file.readline().split(" ")[1])
            nbEdges = int(file.readline().split(" ")[1])
            
            adjacencyMatrix, file = readEdges(file, nbNodes, nbEdges)
            
        elif "SECTION Terminals" in line:
            nbTerminals = int(file.readline().split(" ")[1])
            
            terminals, file = readTerminals(file, nbTerminals)
           
        line = file.readline()

    return adjacencyMatrix, terminals
        
readInstance(os.getcwd()+"\heuristic\instance001.gr")