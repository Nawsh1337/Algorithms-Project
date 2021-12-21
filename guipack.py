import tkinter
from tkinter import *
from tkinter import Tk
import matplotlib.pyplot as plt
from tkinter import filedialog
import numpy
import networkx as nx
import numpy
import sys
from collections import defaultdict
from matplotlib.cm import ScalarMappable

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
    global path_text,text2
    text2.configure(state='normal')
    text2.delete('1.0', END)
    text2.insert(tkinter.END,"No Output Detected Yet", 'color')
    text2.configure(state='disabled')
    if algochoose.get()=="Original":
        plotter()
        plt.title("Original Graph")
        plt.show()
    elif algochoose.get()=="Prim's":
        new_edges = prims(num_nodes, edges)
        prims_plotter(new_edges)
        plt.title("Prim's Graph")
        text2.configure(state='normal')
        text2.delete('1.0', END)
        text2.insert(tkinter.END, path_text, 'color')
        text2.configure(state='disabled')
        path_text = ""
        plt.show()
    elif algochoose.get()=="Kruskal's":
        g = kruskal(num_nodes, edges)
        new_edges=g.kruskals_algorithm()
        kruskals_plotter(new_edges)
        plt.title("Kruskal's Graph")
        text2.configure(state='normal')
        text2.delete('1.0', END)
        text2.insert(tkinter.END, path_text, 'color')
        text2.configure(state='disabled')
        path_text = ""
        plt.show()
    elif algochoose.get() == "Dijkstra's":
        g = dijkstra()
        new_edges=g.dijkstras_algorithm(start_node,num_nodes,edges)
        dijkstras_plotter(new_edges)
        plt.title("Dijkstra's Graph")
        text2.configure(state='normal')
        text2.delete('1.0', END)
        text2.insert(tkinter.END, path_text, 'color')
        text2.configure(state='disabled')
        path_text = ""
        plt.show()
    elif algochoose.get() == "Bellman Ford":
        new_edges = Bellman_ford(start_node, num_nodes, len(edges), edges)
        bellman_ford_plotter(new_edges)
        plt.title("Bellman Ford Graph")
        text2.configure(state='normal')
        text2.delete('1.0', END)
        text2.insert(tkinter.END, path_text, 'color')
        text2.configure(state='disabled')
        path_text = ""
        plt.show()
    elif algochoose.get() == "Floyd Warshall":
        g = floyd_warshal(num_nodes, edges)
        new_edges=g.floyd_warshall_algorithm()
        floyd_warshal_plotter(new_edges)
        plt.title("Floyd Warshall Graph")
        text2.configure(state='normal')
        text2.delete('1.0', END)
        text2.insert(tkinter.END, path_text, 'color')
        text2.configure(state='disabled')
        path_text = ""
        plt.show()
    elif algochoose.get() == "Boruvka's":
        g = boruvka(num_nodes, edges)
        new_edges=g.boruvka_mst()
        boruvka_plotter(new_edges)
        plt.title("Boruvka's Graph")
        text2.configure(state='normal')
        text2.delete('1.0', END)
        text2.insert(tkinter.END, path_text, 'color')
        text2.configure(state='disabled')
        path_text = ""
        plt.show()
    elif algochoose.get() == "Local Clustering":
        local_clustering(num_nodes, nodes, edges)
        plt.title("Local Clustering Graph")
        text2.configure(state='normal')
        text2.delete('1.0', END)
        text2.insert(tkinter.END, path_text, 'color')
        text2.configure(state='disabled')
        path_text = ""
        plt.show()
    file_parser_after_draw()
def file_parser_after_draw():
    global edges,num_nodes,start_node
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
    num_nodes = int(y[0][0])
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
        b.append((edges[i][0], edges[i][1], float(edges[i][2] / 10000000)))
    edges.clear()
    b.sort()

    for i in range(len(b)):
        for j in range(i, len(b)):
            if (b[i][0] == b[j][1] and b[i][1] == b[j][0] and b[i][2] < b[j][2]):
                edges.append(b[i])
                break
            elif (b[i][0] == b[j][1] and b[i][1] == b[j][0] and b[i][2] > b[j][2]):
                edges.append(b[j])
                break
            elif (b[i][0] == b[j][1] and b[i][1] == b[j][0] and b[i][2] == b[j][2]):
                edges.append(b[i])
                edges.append(b[j])
                break
def file_parser():
    drop.config(state='normal')
    global nodes, edges, a, b, path, x, num_nodes, start_node
    nodes = []
    edges = []
    a = []
    b = []
    test = tkinter.Tk()
    test.withdraw()

    path = filedialog.askopenfilename()

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
    num_nodes = int(y[0][0])
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
        b.append((edges[i][0], edges[i][1], float(edges[i][2] / 10000000)))
    edges.clear()
    b.sort()

    for i in range(len(b)):
        for j in range(i, len(b)):
            if (b[i][0] == b[j][1] and b[i][1] == b[j][0] and b[i][2] < b[j][2]):
                edges.append(b[i])
                break
            elif (b[i][0] == b[j][1] and b[i][1] == b[j][0] and b[i][2] > b[j][2]):
                edges.append(b[j])
                break
            elif (b[i][0] == b[j][1] and b[i][1] == b[j][0] and b[i][2] == b[j][2]):
                edges.append(b[i])
                edges.append(b[j])
                break
    # parser done till here
    # code to plot on math.lib
def prims(V, G):
    global path_text
    path_text = ""
    adjMatrix = []
    adjMatrix.clear()
    for i in range(0, V):
        adjMatrix.append([])
        for j in range(0, V):
            adjMatrix[i].append(0)

    for i in range(0, len(G)):
        adjMatrix[G[i][0]][G[i][1]] = G[i][2]
        adjMatrix[G[i][1]][G[i][0]] = G[i][2]

    vertex = 0
    MST = []
    MST.clear()
    edges = []
    edges.clear()
    visited = []
    visited.clear()
    minEdge = [None, None, float('inf')]

    while len(MST) != V - 1:
        visited.append(vertex)
        for r in range(0, V):
            if adjMatrix[vertex][r] != 0:
                edges.append([vertex, r, adjMatrix[vertex][r]])

        for e in range(0, len(edges)):
            if edges[e][2] < minEdge[2] and edges[e][1] not in visited:
                minEdge = edges[e]

        edges.remove(minEdge)
        MST.append(minEdge)
        vertex = minEdge[1]
        minEdge = [None, None, float('inf')]
        MSTweight = 0
    for i in range(len(MST)):
        path_text += "Edge " + str(MST[i][0]) + "-" + str(MST[i][1]) + " with weight " + str(MST[i][2]) + " included in MST\n"
        MSTweight = MSTweight + MST[i][2]
    path_text+="Weight of MST is " + str(MSTweight) +'\n'
    return MST
def prims_plotter(edges):# for original
    g = nx.Graph()
    for i in range(num_nodes):
        g.add_node(i, pos=nodes[i][1:3])
    for i in range(len(edges)):
        g.add_edge(edges[i][0], edges[i][1], weight=edges[i][2])

    fig, ax = plt.subplots()
    pos = nx.get_node_attributes(g, 'pos')
    weight = nx.get_edge_attributes(g, 'weight')
    nx.draw(g, pos, with_labels=1, node_size=200, width=1, edge_color="y", node_color="red", ax=ax)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=weight, font_size=6, font_family="sans-serif")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.axis('on')
class kruskal:
    def __init__(self, vertex, e):
        self.V = vertex
        self.graph = []
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
        global path_text
        path_text= ""
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

        # print of krsukal result
        MSTweight = 0
        for i in range(len(result)):
            MSTweight = MSTweight + result[i][2]
            path_text += "Edge " + str(result[i][0]) + "-" + str(result[i][1]) + " with weight " + str(result[i][2]) + " included in MST\n"
        path_text += "Weight of MST is " + str(MSTweight)
        return result
def kruskals_plotter(edges):# for original
    g = nx.Graph()
    for i in range(num_nodes):
        g.add_node(i, pos=nodes[i][1:3])
    for i in range(len(edges)):
        g.add_edge(edges[i][0], edges[i][1], weight=edges[i][2])

    fig, ax = plt.subplots()
    pos = nx.get_node_attributes(g, 'pos')
    weight = nx.get_edge_attributes(g, 'weight')
    nx.draw(g, pos, with_labels=1, node_size=200, width=1, edge_color="y", node_color="red", ax=ax)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=weight, font_size=6, font_family="sans-serif")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.axis('on')
def plotter():# for original
    graph = nx.DiGraph()
    for i in range(num_nodes):
        graph.add_node(i, pos=nodes[i][1:3])
    for i in range(len(edges)):
        graph.add_edge(edges[i][0], edges[i][1], weight=edges[i][2])

    fig, ax = plt.subplots()
    pos = nx.get_node_attributes(graph, 'pos')
    weight = nx.get_edge_attributes(graph, 'weight')
    nx.draw(graph, pos, with_labels=1, node_size=200, width=1, edge_color="y", node_color="red", ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weight, font_size=6, font_family="sans-serif")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.axis('on')
class dijkstra:
    dij_edges = []
    global path_text
    path_text = ""
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
        global path_text
        if parent[j] == -1:
            path_text += str(j)
            path_text += '\n'
            return
        self.printPath(parent, parent[j])
        self.dij_edges.append(j)
        path_text += str(j)
        path_text += '\n'

    def printSolution(self, dist, parent, src):
        global path_text
        path_text = ""
        path_text += "Vertex \t\tDistance from Source"
        path_text += '\n'
        for i in range(0, len(dist)):
            path_text += '\n' + str(src) + '-->' + str(i) + '\t\t' + str(dist[i]) + '\n'
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
        self.dij_edges.clear()
        return c
def dijkstras_plotter(edges):# for original
    graph = nx.DiGraph()
    for i in range(num_nodes):
        graph.add_node(i, pos=nodes[i][1:3])
    for i in range(len(edges)):
        graph.add_edge(edges[i][0], edges[i][1], weight=edges[i][2])

    fig, ax = plt.subplots()
    pos = nx.get_node_attributes(graph, 'pos')
    weight = nx.get_edge_attributes(graph, 'weight')
    nx.draw(graph, pos, with_labels=1, node_size=200, width=1, edge_color="y", node_color="red", ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weight, font_size=6, font_family="sans-serif")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.axis('on')
def Bellman_ford(src, n, m, graph):
    global path_text
    path_text = ""
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
            path_text += "Graph contains negative weight cycle\n"
            cycle = 1
            break
    if cycle == 0:
        path_text += "Distance from source vertex,  " + str(src) + "\n"
        path_text += 'Vertex \t Distance from source\n'
        for i in range(len(dist)):
            path_text += str(i) + '\t' + str(dist[i]) +"\n"

    return (edges)
def bellman_ford_plotter(edges):
    graph = nx.DiGraph()
    for i in range(num_nodes):
        graph.add_node(i, pos=nodes[i][1:3])
    for i in range(len(edges)):
        graph.add_edge(edges[i][0], edges[i][1], weight=edges[i][2])

    fig, ax = plt.subplots()
    pos = nx.get_node_attributes(graph, 'pos')
    weight = nx.get_edge_attributes(graph, 'weight')
    nx.draw(graph, pos, with_labels=1, node_size=200, width=1, edge_color="y", node_color="red", ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weight, font_size=6, font_family="sans-serif")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.axis('on')
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

    def print_floyd_warshal(self):
        path = []
        global path_text
        path_text=""
        for i in range(self.V):
            for j in range(self.V):
                if i != j:
                    path = self.constructPath(i, j)
                    self.printPath(path)
                    path = []

    def printPath(self, path):
        global path_text
        n = len(path)
        for i in range(n - 1):
            path_text += str(path[i]) + " -> "
        path_text += str(path[n-1]) + "\n"

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
    graph = nx.DiGraph()
    for i in range(num_nodes):
        graph.add_node(i, pos=nodes[i][1:3])
    for i in range(len(edges)):
        graph.add_edge(edges[i][0], edges[i][1], weight=edges[i][2])

    fig, ax = plt.subplots()
    pos = nx.get_node_attributes(graph, 'pos')
    weight = nx.get_edge_attributes(graph, 'weight')
    nx.draw(graph, pos, with_labels=1, node_size=200, width=1, edge_color="y", node_color="red", ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weight, font_size=6, font_family="sans-serif")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.axis('on')
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
        global path_text
        path_text = ""
        for i in range(len(edges)):
            path_text += "Edge " + str(edges[i][0]) + "-" + str(edges[i][1]) + " with weight " + str(edges[i][2]) + " included in MST\n"
        path_text += "Weight of MST is " + str(self.MSTweight)

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
    g = nx.Graph()
    for i in range(num_nodes):
        g.add_node(i, pos=nodes[i][1:3])
    for i in range(len(edges)):
        g.add_edge(edges[i][0], edges[i][1], weight=edges[i][2])

    fig, ax = plt.subplots()
    pos = nx.get_node_attributes(g, 'pos')
    weight = nx.get_edge_attributes(g, 'weight')
    nx.draw(g, pos, with_labels=1, node_size=200, width=1, edge_color="y", node_color="red", ax=ax)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=weight, font_size=6, font_family="sans-serif")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.axis('on')
def local_clustering(num_nodes, nodes, edges):
    average = 0.0
    global path_text
    path_text = ""
    sum = 0
    g = nx.Graph()
    for i, j, k in edges:
        g.add_edge(i, j)
    for i in range(num_nodes):
        path_text += "clustering of "+ str(i) + " is: "
        sum += nx.clustering(g, i)
        path_text += str(nx.clustering(g, i)) + '\n'
    average = sum / num_nodes
    path_text += "average local clustering of graph is: " + str(average) +'\n'
    gc = g.subgraph(max(nx.connected_components(g)))
    lcc = nx.clustering(gc)

    cmap = plt.get_cmap('autumn')
    norm = plt.Normalize(0, max(lcc.values()))
    node_colors = [cmap(norm(lcc[node])) for node in gc.nodes]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
    nx.draw_spring(gc, node_color=node_colors, with_labels=True, ax=ax1)
    fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), label='Clustering', shrink=0.95, ax=ax1)

    ax2.hist(lcc.values(), bins=10)
    ax2.set_xlabel('Clustering')
    ax2.set_ylabel('Frequency')
    plt.tight_layout()
def algoselected(event):
    draw_button.config(state='normal')
#create windows form
app= Tk()
def destroy_initial():
    begin_button.destroy()
    begin_label.destroy()
    # dynamically resize
    Grid.rowconfigure(app, index=0, weight=1)
    Grid.columnconfigure(app, index=0, weight=1)
    Grid.columnconfigure(app, index=1, weight=1)
    Grid.columnconfigure(app, index=2, weight=1)
    Grid.rowconfigure(app, index=2, weight=1)
    Grid.rowconfigure(app, index=3, weight=1)
    Grid.rowconfigure(app, index=4, weight=1)
    global filechoose_image
    filechoose_image = tkinter.PhotoImage(file='images/fileselect.png')
    filechoose = tkinter.Label(app, text='Choose File', font=('Ubuntu',12,'bold'), bg='blue', fg='black')
    filechoose.grid(row=0, column=0, sticky="se", padx=20)

    filechoose_button = tkinter.Button(app, image=filechoose_image, width=20, command=file_parser, borderwidth=0)
    filechoose_button.grid(row=1, column=0, sticky="e", padx=20, pady=20)
    filechoose_button.bind("<Enter>", on_enter)
    filechoose_button.bind("<Leave>", on_leave)

    algochoose_text = tkinter.Label(app, text='Choose Algorithm', font=('Ubuntu',12,'bold'), bg='blue', fg='black')
    algochoose_text.grid(row=0, column=1, sticky='s')
    options = ["Original", "Prim's", "Kruskal's", "Dijkstra's", "Bellman Ford", "Floyd Warshall", "Boruvka's",
               "Local Clustering"]
    global algochoose
    algochoose = StringVar()
    algochoose.set(options[0])
    global drop
    drop = tkinter.OptionMenu(app, algochoose, *options, command=algoselected)
    drop.config(bg='lightblue',font=('Ubuntu',10,'bold'))
    drop.grid(row=1, column=1, sticky='n', pady=20)
    drop.config(state='disabled')

    endprog = tkinter.Button(app, text='End Program', command=exiter,font=('Ubuntu',10,'bold'), bg='white', fg='black')
    endprog.grid(row=0, column=2, sticky='sw')
    endprog.bind("<Enter>", on_enter_endprog)
    endprog.bind("<Leave>", on_leave_endprog)

    global draw_button
    draw_button = tkinter.Button(app, text="Draw", width=12, command=draw,font=('Ubuntu',10,'bold'), bg='white', fg='black')
    draw_button.grid(row=1, column=2, sticky='nw', pady=20)
    draw_button.bind("<Enter>", on_enter)
    draw_button.bind("<Leave>", on_leave)
    draw_button.config(state='disabled')

    path_label = tkinter.Label(app, text='Path For Graph', font=('Ubuntu',12,'bold'), bg='blue', fg='black')
    path_label.grid(row=3, column=1, sticky='n')

    global text2#textbox to print path
    text2 = tkinter.Text(app, height=6, width=70,bg="#78818f")
    scroll = tkinter.Scrollbar(app, command=text2.yview)
    text2.configure(yscrollcommand=scroll.set)
    text2.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
    text2.tag_configure('big', font=('Verdana', 20, 'bold'))
    text2.tag_configure('color', foreground='black', font=('Arial', 10, 'bold'))
    global path_text
    path_text = StringVar()  # display path
    path_text="No Output Detected Yet"
    text2.insert(tkinter.END, path_text, 'color')
    text2.grid(row=3, column=1, sticky='s')
    text2.configure(state='disabled')

app.title('Graph Visualizer')
app.geometry('800x350')
app.config(bg='#609da3')
begin_label = tkinter.Label(app,text='Graph Visualizer Application',font=('Ubuntu',44,'bold'),bg='purple',fg='black')
begin_label.grid(row=0,column=0,sticky='news')


begin_button = tkinter.Button(app, text="Begin",font=('Ubuntu',30,'bold'),width=20,command=destroy_initial,borderwidth=0,bg='grey')
begin_button.grid(row=1,column=0, sticky='news')
begin_button.bind("<Enter>", on_enter)
begin_button.bind("<Leave>", on_leave)
Grid.rowconfigure(app,index=0,weight=1)
Grid.rowconfigure(app,index=1,weight=1)
Grid.columnconfigure(app,index=0,weight=1)

app= mainloop()
