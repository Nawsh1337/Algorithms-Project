import tkinter
from tkinter import *
from tkinter import Tk
from tkinter.ttk import *
import os
import matplotlib.pyplot as plt
import numpy
import numpy as np
import networkx as nx
from tkinter import filedialog

def on_enter(e):
    e.widget['background'] = 'green'

def on_leave(e):
    e.widget['background'] = 'white'

def on_enter_endprog(e):
    e.widget['background'] = 'red'

def on_leave_endprog(e):
    e.widget['background'] = 'white'

def exiter():
    exit(0)
def draw():
    if algochoose.get()=="Original":
        plotter()
        plt.title("Original Graph")
        plt.show()
    elif algochoose.get()=="Prim's":
        print("im primsssssss")
        new_edges = prims(num_nodes, edges)
        prims_plotter(new_edges)
        plt.title("Prim's Graph")
        plt.show()
    elif algochoose.get()=="Kruskal's":
        print("Im kruskalkllll")
        g = kruskal(num_nodes)
        g.add_edge(edges)
        new_edges=g.kruskals_algorithm()
        kruskals_plotter(nodes,new_edges)
        plt.title("Kruskal's Graph")
        plt.show()
    elif algochoose.get() == "Dijkstra's":
        g = dijkstra()
        new_edges=g.dijkstras_algorithm(start_node,num_nodes,edges)
        dijkstras_plotter(new_edges)
        plt.title("Dijkstra's Graph")
        plt.show()
    elif algochoose.get() == "Bellman Ford":
        new_edges = Bellman_ford(start_node, num_nodes, len(edges), edges)
        bellman_ford_plotter(new_edges)
        plt.title("Bellman Ford Graph")
        plt.show()
    elif algochoose.get() == "Floyd Warshall":
        g = floyd_warshal(num_nodes, edges)
        new_edges=g.floyd_warshall_algorithm()
        floyd_warshal_plotter(new_edges)
        plt.title("Floyd Warshall Graph")
        plt.show()
    elif algochoose.get() == "Boruvka's":
        g = boruvka(num_nodes, edges)
        new_edges=g.boruvka_mst()
        boruvka_plotter(new_edges)
        plt.title("Boruvka's Graph")
        plt.show()
    file_parser_after_draw()
def file_parser_after_draw():
    edges.clear()
    f_read = open(path, "rt")
    x = f_read.read()
    f_read.close()

    x = x.replace("NETSIM", "")
    lines = x.split("\n")
    non_empty_lines = [line.strip() for line in lines if line.strip() != ""]

    string_without_empty_lines = ""
    for line in non_empty_lines:
        string_without_empty_lines += line + "\n"
    x = string_without_empty_lines

    y = x.splitlines()

    y = [i.split("\t") for i in y]  # split string inside each list
    global num_nodes
    num_nodes = int(y[0][0])
    global start_node
    start_node = int(y[-1][0])
    y.pop(0)
    y.pop()
    a = []
    b = []
    for j in range(num_nodes):  # extracted nodes
        a = [float(i) for i in y[0]]
        nodes.append((int(a[0]), a[1], a[2]))
        y.pop(0)
    a.clear()

    for i in range(len(y)):  # removes unnecessary bandwidth
        for j in range(2, len(y[i]), 2):
            a.append(y[i][j])
        for j in range(len(a)):
            for k in range(1, len(y[i])):
                if y[i][k] == a[j]:
                    del y[i][k]
                    break
        a.clear()

    k = 0
    for i in y:  # extract edges
        length = int(i.pop(0))
        for j in range(length):
            edges.append((nodes[k][0], int(i[0]), float(i[1])))
            i.pop(0)
            i.pop(0)
        k += 1

    for i in range(len(edges)):  # convert in integer and remove nonimportant bandwidth
        if edges[i][0] == edges[i][1]:
            continue
        b.append((edges[i][0], edges[i][1], int(edges[i][2] / 1000000)))

    edges.clear()
    for i in b:
        edges.append(i)
    b.clear()
    edges.sort()
def file_parser():
    drop.config(state='normal')
    global nodes
    nodes = []
    global edges
    edges = []
    global a
    global b
    a = []
    b = []
    test = tkinter.Tk()
    test.withdraw()
    global path
    path = filedialog.askopenfilename(initialdir="C:/Users/MainFrame/Desktop/",title="Open Text file",filetypes=(("Text Files Only", "*.txt"),))

    f_read = open(path, "rt")

    global x
    x = f_read.read()
    f_read.close()

    x = x.replace("NETSIM", "")
    lines = x.split("\n")
    non_empty_lines = [line.strip() for line in lines if line.strip() != ""]

    string_without_empty_lines = ""
    for line in non_empty_lines:
        string_without_empty_lines += line + "\n"
    x = string_without_empty_lines

    y = x.splitlines()

    y = [i.split("\t") for i in y]  # split string inside each list
    global num_nodes
    num_nodes = int(y[0][0])
    global start_node
    start_node = int(y[-1][0])
    y.pop(0)
    y.pop()
    a = []
    b = []
    for j in range(num_nodes):  # extracted nodes
        a = [float(i) for i in y[0]]
        nodes.append((int(a[0]), a[1], a[2]))
        y.pop(0)
    a.clear()

    for i in range(len(y)):  # removes unnecessary bandwidth
        for j in range(2, len(y[i]), 2):
            a.append(y[i][j])
        for j in range(len(a)):
            for k in range(1, len(y[i])):
                if y[i][k] == a[j]:
                    del y[i][k]
                    break
        a.clear()

    k = 0
    for i in y:  # extract edges
        length = int(i.pop(0))
        for j in range(length):
            edges.append((nodes[k][0], int(i[0]), float(i[1])))
            i.pop(0)
            i.pop(0)
        k += 1

    for i in range(len(edges)):  # convert in integer and remove nonimportant bandwidth
        if edges[i][0] == edges[i][1]:
            continue
        b.append((edges[i][0], edges[i][1], int(edges[i][2] / 1000000)))

    edges.clear()
    for i in b:
        edges.append(i)
    b.clear()
    edges.sort()
    # parser done till here
    # code to plot on math.lib
def prims(V, G):
    # create adj matrix from graph
    adjMatrix = []
      #create N x N matrix filled with 0 edge weights between all vertices
    for i in range(0, V):
        adjMatrix.append([])
        for j in range(0, V):
            adjMatrix[i].append(0)
      #populate adjacency matrix with correct edge weights
    for i in range(0, len(G)):
        adjMatrix[G[i][0]][G[i][1]] = G[i][2]
        adjMatrix[G[i][1]][G[i][0]] = G[i][2]
    #arbitrarily choose initial vertex from graph
    vertex = 0
    #initialize empty edges array and empty MST
    MST = []
    edges = []
    visited = []
    minEdge = [None,None,float('inf')]
    #run prims algorithm until we create an MST
    #that contains every vertex from the graph
    while len(MST) != V-1:
        #mark this vertex as visited
        visited.append(vertex)
        #add each edge to list of potential edges
        for r in range(0, V):
            if adjMatrix[vertex][r] != 0:
                edges.append([vertex,r,adjMatrix[vertex][r]])
          #find edge with the smallest weight to a vertex
          #that has not yet been visited
        for e in range(0, len(edges)):
            if edges[e][2] < minEdge[2] and edges[e][1] not in visited:
                minEdge = edges[e]
          #remove min weight edge from list of edges
        edges.remove(minEdge)
        #push min edge to MST
        MST.append(minEdge)
        #start at new vertex and reset min edge
        vertex = minEdge[1]
        minEdge = [None,None,float('inf')]
    return MST
def prims_plotter(edges):# for original
    bandwidth = []
    point = []
    e = []
    a = []
    for i in range(len(nodes)):
        point.append(nodes[i][1:3])

    for i in range(len(edges)):
        if edges[i][0] != edges[i][1]:
            a.append(edges[i][0])
            a.append(edges[i][1])
            e.append(a[0:2])
            a.clear()

    for i in range(len(edges)):
        bandwidth.append(edges[i][2])

    # plot on mathplot.lib
    points = numpy.array(point)
    edge = numpy.array(e)

    x = points[:, 0].flatten()
    y = points[:, 1].flatten()

    plt.plot(x[edge.T], y[edge.T], linestyle='-', color='y',
             markerfacecolor='red', marker='o')

    for i in range(len(points)):
        plt.annotate(i, (points[i]))
    edges.sort()
    start = 0
    end = 0
    i = 0
    while i < len(edges) - 1:
        start = edges[i][0]
        end = edges[i][1]
        plt.annotate(bandwidth[i], ((((points[int(edges[i][0])][0]) * 0.75 + (points[int(edges[i][1])][0]) * 0.25),
                                     (((points[int(edges[i][0])][1]) * 0.75 + (
                                         points[int(edges[i][1])][1]) * 0.25)))))
        while (1):
            if (start == edges[i + 1][0] and end == edges[i + 1][1]):
                i += 1
            else:
                i += 1
                break

    plt.title("Here we change it to dijkstras etc. according to given algo")
class kruskal:
    def __init__(self, vertex):
        self.V = vertex
        self.graph = []

    def add_edge(self, e):
        self.graph = e

    def search(self, parent, i):
        if parent[i] == i:
            return i
        return self.search(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.search(parent, x)
        yroot = self.search(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskals_algorithm(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.search(parent, u)
            y = self.search(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        # for u, v, weight in result:
        #    print("Edge:",u, v,end =" ")
        #    print("-",weight)
        return result
def kruskals_plotter(nodes,edges):# for original
    bandwidth = []
    point = []
    e = []
    a = []
    for i in range(len(nodes)):
        point.append(nodes[i][1:3])

    for i in range(len(edges)):
        if edges[i][0] != edges[i][1]:
            a.append(edges[i][0])
            a.append(edges[i][1])
            e.append(a[0:2])
            a.clear()

    for i in range(len(edges)):
        bandwidth.append(edges[i][2])

    # plot on mathplot.lib
    points = numpy.array(point)
    edge = numpy.array(e)

    x = points[:, 0].flatten()
    y = points[:, 1].flatten()

    plt.plot(x[edge.T], y[edge.T], linestyle='-', color='y',
             markerfacecolor='red', marker='o')

    for i in range(len(points)):
        plt.annotate(i, (points[i]))
    edges.sort()
    start = 0
    end = 0
    i = 0
    while i < len(edges) - 1:
        start = edges[i][0]
        end = edges[i][1]
        plt.annotate(bandwidth[i], ((((points[int(edges[i][0])][0]) * 0.75 + (points[int(edges[i][1])][0]) * 0.25),
                                     (((points[int(edges[i][0])][1]) * 0.75 + (
                                         points[int(edges[i][1])][1]) * 0.25)))))
        while (1):
            if (start == edges[i + 1][0] and end == edges[i + 1][1]):
                i += 1
            else:
                i += 1
                break

    plt.title("Here we change it to dijkstras etc. according to given algo")
def plotter():# for original
    bandwidth = []
    point = []
    e = []
    a = []
    for i in range(len(nodes)):
        point.append(nodes[i][1:3])

    for i in range(len(edges)):
        if edges[i][0] != edges[i][1]:
            a.append(edges[i][0])
            a.append(edges[i][1])
            e.append(a[0:2])
            a.clear()

    for i in range(len(edges)):
        bandwidth.append(edges[i][2])

    # plot on mathplot.lib
    points = numpy.array(point)
    edge = numpy.array(e)

    x = points[:, 0].flatten()
    y = points[:, 1].flatten()

    plt.plot(x[edge.T], y[edge.T], linestyle='-', color='y',
             markerfacecolor='red', marker='o')

    for i in range(len(points)):
        plt.annotate(i, (points[i]))
    edges.sort()
    start = 0
    end = 0
    i = 0
    while i < len(edges) - 1:
        start = edges[i][0]
        end = edges[i][1]
        plt.annotate(bandwidth[i], ((((points[int(edges[i][0])][0]) * 0.75 + (points[int(edges[i][1])][0]) * 0.25),
                                     (((points[int(edges[i][0])][1]) * 0.75 + (
                                         points[int(edges[i][1])][1]) * 0.25)))))
        while (1):
            if (start == edges[i + 1][0] and end == edges[i + 1][1]):
                i += 1
            else:
                i += 1
                break

    plt.title("Here we change it to dijkstras etc. according to given algo")
class dijkstra:
    dij_edges = []

    def minDistance(self, dist, queue):
        # Initialize min value and min_index as -1
        minimum = float("Inf")
        min_index = -1

        # from the dist array,pick one which
        # has min value and is till in queue
        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i
        return min_index

    def printPath(self, parent, j):
        if parent[j] == -1:
            print(j)
            return
        self.printPath(parent, parent[j])
        self.dij_edges.append(j)
        print(j)

    def printSolution(self, dist, parent, src):
        print("Vertex \t\tDistance from Source\tPath")
        for i in range(0, len(dist)):
            print("\n%d --> %d \t\t%d \t\t\t\t\t" % (src, i, dist[i])),
            self.dij_edges.append(src)
            self.printPath(parent, i)

    def dijkstras_algorithm(self, src, V, G):
        adjMatrix = []

        for i in range(0, V):
            adjMatrix.append([])
            for j in range(0, V):
                adjMatrix[i].append(0)

        for i in range(0, len(G)):
            adjMatrix[G[i][0]][G[i][1]] = G[i][2]
            adjMatrix[G[i][1]][G[i][0]] = G[i][2]
        graph = adjMatrix

        row = len(graph)
        col = len(graph[0])

        dist = [float("Inf")] * row

        parent = [-1] * row

        dist[src] = 0

        queue = []
        for i in range(row):
            queue.append(i)

        while queue:
            u = self.minDistance(dist, queue)
            queue.remove(u)
            for i in range(col):
                if graph[u][i] and i in queue:
                    if dist[u] + graph[u][i] < dist[i]:
                        dist[i] = dist[u] + graph[u][i]
                        parent[i] = u

        self.printSolution(dist, parent, src)
        b = []
        c = []
        for j in range(len(self.dij_edges) - 1):
            if self.dij_edges[j + 1] == src:
                j += 2
            else:
                b.append((self.dij_edges[j], self.dij_edges[j + 1]))

        for i in b:
            if i not in c:
                c.append(i)

        for i in range(len(c)):
            for j in range(len(G)):
                if c[i] == G[j][0:2]:
                    c[i] = G[j]
        return c
def dijkstras_plotter(edges):# for original
    bandwidth = []
    point = []
    e = []
    a = []
    for i in range(len(nodes)):
        point.append(nodes[i][1:3])

    for i in range(len(edges)):
        if edges[i][0] != edges[i][1]:
            a.append(edges[i][0])
            a.append(edges[i][1])
            e.append(a[0:2])
            a.clear()

    for i in range(len(edges)):
        bandwidth.append(edges[i][2])

    # plot on mathplot.lib
    points = numpy.array(point)
    edge = numpy.array(e)

    x = points[:, 0].flatten()
    y = points[:, 1].flatten()

    plt.plot(x[edge.T], y[edge.T], linestyle='-', color='y',
             markerfacecolor='red', marker='o')

    for i in range(len(points)):
        plt.annotate(i, (points[i]))
    edges.sort()
    start = 0
    end = 0
    i = 0
    while i < len(edges) - 1:
        start = edges[i][0]
        end = edges[i][1]
        plt.annotate(bandwidth[i], ((((points[int(edges[i][0])][0]) * 0.75 + (points[int(edges[i][1])][0]) * 0.25),
                                     (((points[int(edges[i][0])][1]) * 0.75 + (
                                         points[int(edges[i][1])][1]) * 0.25)))))
        while (1):
            if (start == edges[i + 1][0] and end == edges[i + 1][1]):
                i += 1
            else:
                i += 1
                break

    plt.title("Here we change it to dijkstras etc. according to given algo")
def Bellman_ford(src, n, m, graph):
    dist = [float("inf") for i in range(n)]
    dist[src] = 0
    edges = []
    for i in range(n - 1):
        for u, v, w in graph:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                edges.append((u, v, w))

    cycle = 0
    for u, v, w in graph:
        if dist[u] != float("Inf") and dist[u] + w < dist[v]:
            print("Graph contains negative weight cycle")
            cycle = 1
            break
    if cycle == 0:
        print('Distance from source vertex', src)
        print('Vertex \t Distance from source')
        for i in range(len(dist)):
            print(i, '\t', dist[i])

    return (edges)
def bellman_ford_plotter(edges):
    bandwidth = []
    point = []
    e = []
    a = []
    for i in range(len(nodes)):
        point.append(nodes[i][1:3])

    for i in range(len(edges)):
        if edges[i][0] != edges[i][1]:
            a.append(edges[i][0])
            a.append(edges[i][1])
            e.append(a[0:2])
            a.clear()

    for i in range(len(edges)):
        bandwidth.append(edges[i][2])

    # plot on mathplot.lib
    points = numpy.array(point)
    edge = numpy.array(e)

    x = points[:, 0].flatten()
    y = points[:, 1].flatten()

    plt.plot(x[edge.T], y[edge.T], linestyle='-', color='y',
             markerfacecolor='red', marker='o')

    for i in range(len(points)):
        plt.annotate(i, (points[i]))
    edges.sort()
    start = 0
    end = 0
    i = 0
    while i < len(edges) - 1:
        start = edges[i][0]
        end = edges[i][1]
        plt.annotate(bandwidth[i], ((((points[int(edges[i][0])][0]) * 0.75 + (points[int(edges[i][1])][0]) * 0.25),
                                     (((points[int(edges[i][0])][1]) * 0.75 + (
                                         points[int(edges[i][1])][1]) * 0.25)))))
        while (1):
            if (start == edges[i + 1][0] and end == edges[i + 1][1]):
                i += 1
            else:
                i += 1
                break
class floyd_warshal:

    def __init__(self, vertex, G):
        self.edges = G
        self.V = vertex
        self.graph = []
        self.MAXM, self.INF = 100, 10 ** 7
        self.dis = [[-1 for i in range(self.MAXM)] for i in range(self.MAXM)]
        self.Next = [[-1 for i in range(self.MAXM)] for i in range(self.MAXM)]
        for i in range(0, self.V):
            self.graph.append([])
            for j in range(0, self.V):
                self.graph[i].append(0)
        for i in range(0, len(G)):
            self.graph[G[i][0]][G[i][1]] = G[i][2]
            self.graph[G[i][1]][G[i][0]] = G[i][2]

        for i in range(self.V):
            for j in range(self.V):
                if i != j and self.graph[i][j] == 0:
                    self.graph[i][j] = self.INF

        for i in range(self.V):
            for j in range(self.V):
                self.dis[i][j] = self.graph[i][j]
                if (self.graph[i][j] == self.INF):
                    self.Next[i][j] = -1
                else:
                    self.Next[i][j] = j

    def constructPath(self, u, v):
        if (self.Next[u][v] == -1):
            return {}

        path = [u]
        while (u != v):
            u = self.Next[u][v]
            path.append(u)
        return path

    # print results of floyd warshal
    def print_floyd_warshal(self):
        path = []
        for i in range(self.V):
            for j in range(self.V):
                if i != j:
                    path = self.constructPath(i, j)
                    self.printPath(path)
                    path = []

    def printPath(self, path):
        n = len(path)
        for i in range(n - 1):
            print(path[i], end=" -> ")
        print(path[n - 1])

    def floyd_warshall_algorithm(self):
        edges = []
        a = []
        for k in range(self.V):
            for i in range(self.V):
                for j in range(self.V):
                    if (self.dis[i][k] == self.INF or self.dis[k][j] == self.INF):
                        continue
                    if (self.dis[i][j] > self.dis[i][k] + self.dis[k][j]):
                        self.dis[i][j] = self.dis[i][k] + self.dis[k][j]
                        self.Next[i][j] = self.Next[i][k]

        for i in self.edges:
            path = []
            path = self.constructPath(i[0], i[1])
            for j in range(len(path) - 1):
                a.append((path[j], path[j + 1]))
        for i in a:
            if i not in edges:
                edges.append(i)
        path = []
        for i in self.edges:
            if i[0:2] in edges:
                path.append(i)
        self.print_floyd_warshal()
        return path
def floyd_warshal_plotter(edges):
    bandwidth = []
    point = []
    e = []
    a = []
    for i in range(len(nodes)):
        point.append(nodes[i][1:3])

    for i in range(len(edges)):
        if edges[i][0] != edges[i][1]:
            a.append(edges[i][0])
            a.append(edges[i][1])
            e.append(a[0:2])
            a.clear()

    for i in range(len(edges)):
        bandwidth.append(edges[i][2])

    # plot on mathplot.lib
    points = numpy.array(point)
    edge = numpy.array(e)

    x = points[:, 0].flatten()
    y = points[:, 1].flatten()

    plt.plot(x[edge.T], y[edge.T], linestyle='-', color='y',
             markerfacecolor='red', marker='o')

    for i in range(len(points)):
        plt.annotate(i, (points[i]))
    edges.sort()
    start = 0
    end = 0
    i = 0
    while i < len(edges) - 1:
        start = edges[i][0]
        end = edges[i][1]
        plt.annotate(bandwidth[i], ((((points[int(edges[i][0])][0]) * 0.75 + (points[int(edges[i][1])][0]) * 0.25),
                                     (((points[int(edges[i][0])][1]) * 0.75 + (
                                         points[int(edges[i][1])][1]) * 0.25)))))
        while (1):
            if (start == edges[i + 1][0] and end == edges[i + 1][1]):
                i += 1
            else:
                i += 1
                break

    plt.title("Here we change it to dijkstras etc. according to given algo")
class boruvka:

    def __init__(self, vertices, e):
        self.V = vertices
        self.graph = []
        self.MSTweight = 0
        self.graph = e

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # print boruvkas resut

    def print_boruvkas(self, edges):
        for i in range(len(edges)):
            print("Edge %d-%d with weight %d included in MST" % (edges[i][0], edges[i][1], edges[i][2]))
        print("Weight of MST is %d" % self.MSTweight)

    def boruvka_mst(self):
        parent = []
        rank = []
        edges = []
        cheapest = []
        numTrees = self.V

        for node in range(self.V):
            parent.append(node)
            rank.append(0)
            cheapest = [-1] * self.V

        while numTrees > 1:

            for i in range(len(self.graph)):
                u, v, w = self.graph[i]
                set1 = self.find(parent, u)
                set2 = self.find(parent, v)
                if set1 != set2:

                    if cheapest[set1] == -1 or cheapest[set1][2] > w:
                        cheapest[set1] = [u, v, w]

                    if cheapest[set2] == -1 or cheapest[set2][2] > w:
                        cheapest[set2] = [u, v, w]

            for node in range(self.V):

                if cheapest[node] != -1:
                    u, v, w = cheapest[node]
                    set1 = self.find(parent, u)
                    set2 = self.find(parent, v)

                    if set1 != set2:
                        self.MSTweight += w
                        self.union(parent, rank, set1, set2)
                        edges.append((u, v, w))
                        numTrees = numTrees - 1

            cheapest = [-1] * self.V
        self.print_boruvkas(edges)
        return edges
def boruvka_plotter(edges):
    bandwidth = []
    point = []
    e = []
    a = []
    for i in range(len(nodes)):
        point.append(nodes[i][1:3])

    for i in range(len(edges)):
        if edges[i][0] != edges[i][1]:
            a.append(edges[i][0])
            a.append(edges[i][1])
            e.append(a[0:2])
            a.clear()

    for i in range(len(edges)):
        bandwidth.append(edges[i][2])

    # plot on mathplot.lib
    points = numpy.array(point)
    edge = numpy.array(e)

    x = points[:, 0].flatten()
    y = points[:, 1].flatten()

    plt.plot(x[edge.T], y[edge.T], linestyle='-', color='y',
             markerfacecolor='red', marker='o')

    for i in range(len(points)):
        plt.annotate(i, (points[i]))
    edges.sort()
    start = 0
    end = 0
    i = 0
    while i < len(edges) - 1:
        start = edges[i][0]
        end = edges[i][1]
        plt.annotate(bandwidth[i], ((((points[int(edges[i][0])][0]) * 0.75 + (points[int(edges[i][1])][0]) * 0.25),
                                     (((points[int(edges[i][0])][1]) * 0.75 + (
                                         points[int(edges[i][1])][1]) * 0.25)))))
        while (1):
            if (start == edges[i + 1][0] and end == edges[i + 1][1]):
                i += 1
            else:
                i += 1
                break
def algoselected(event):
    draw_button.config(state='normal')
    print(algochoose.get())
#create windows form
app= Tk()

app.title('Graph Visualizer')
app.geometry('450x350')
app.config(bg='#100f12')

#dynamically resize
Grid.rowconfigure(app,index=0,weight=1)
Grid.columnconfigure(app,index=0,weight=1)
Grid.columnconfigure(app,index=1,weight=1)
Grid.columnconfigure(app,index=2,weight=1)
Grid.rowconfigure(app,index=2,weight=1)

filechoose_image = tkinter.PhotoImage(file='images/fileselect.png')
filechoose = tkinter.Label(app,text='Choose File',font=('bold',12),bg='blue',fg='white')
filechoose.grid(row=0,column=0,sticky="se",padx=20)

filechoose_button = tkinter.Button(app, image=filechoose_image,width=20,command=file_parser,borderwidth=0)
filechoose_button.grid(row=1,column=0,sticky="e",padx=20,pady=20)
filechoose_button.bind("<Enter>", on_enter)
filechoose_button.bind("<Leave>", on_leave)


algochoose_text = tkinter.Label(app, text='Choose Algorithm', font=('bold', 12),bg='blue',fg='white')
algochoose_text.grid(row=0, column=1,sticky='s')
options = ["Original","Prim's", "Kruskal's", "Dijkstra's", "Bellman Ford","Floyd Warshall", "Boruvka's", "Local Clustering"]

algochoose = StringVar()
algochoose.set(options[0])
drop = tkinter.OptionMenu(app, algochoose, *options,command=algoselected)
drop.config(bg='lightgreen')
drop.grid(row=1, column=1,sticky='n',pady=20)
drop.config(state='disabled')


endprog = tkinter.Button(app, text = 'End Program', command = exiter,bg='white',fg='black')
endprog.grid(row = 0, column = 2,sticky='sw')
endprog.bind("<Enter>", on_enter_endprog)
endprog.bind("<Leave>", on_leave_endprog)

draw_button = tkinter.Button(app, text ="Draw",width=12,command=draw,bg='white',fg='black')
draw_button.grid(row=1,column=2,sticky='nw',pady=20)
draw_button.bind("<Enter>", on_enter)
draw_button.bind("<Leave>", on_leave)
draw_button.config(state='disabled')


app= mainloop()
