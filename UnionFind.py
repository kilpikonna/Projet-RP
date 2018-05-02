'''
Created on 10. 4. 2018

@author: Magdalena
'''

class UnionFind(object):
    '''
    classdocs
    '''


    def __init__(self, adjacencyMatrix):
        '''
        Constructor
        '''
#        self.parent = [None for i in range(adjacencyMatrix.shape[0])]
        self.parent = [None for i in range(len(adjacencyMatrix))]
        self.rang = [-1 for i in range(adjacencyMatrix.shape[0])]
    
    def makeSet(self,x):
        self.parent[int(x)] = int(x)
        self.rang[int(x)] = int(x)
        
    def find(self,x):
        if self.parent[int(x)] != int(x):
            self.parent[int(x)] = self.find(self.parent[int(x)])
        return self.parent[int(x)]
    
    def union(self,x, y):
        xRoot = self.find(x)
        yRoot = self.find(y)
        
        if int(xRoot) != int(yRoot):
            if self.rang[int(xRoot)] < self.rang[int(yRoot)]:
                self.parent[int(xRoot)] = int(yRoot)
            else:
                self.parent[int(yRoot)] = int(xRoot)
                if self.rang[int(xRoot)] == self.rang[int(yRoot)]:
                    self.rang[int(xRoot)] = self.rang[int(xRoot)] + 1
                