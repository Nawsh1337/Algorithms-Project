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
        plt.title("Here we change it to dijkstras etc. according to given algo")
        plt.show()
    elif algochoose.get()=="Prim's":
        print("im primsssssss")
        new_edges = prims(num_nodes, edges)
        prims_plotter(new_edges)
        plt.title("Here we change it to dijkstras etc. according to given algo")
        plt.show()
    elif algochoose.get()=="Kruskal's":
        print("Im kruskalkllll")
        g = kruskal(num_nodes)
        g.add_edge(edges)
        new_edges=g.kruskals_algorithm()
        kruskals_plotter(nodes,new_edges)
        plt.title("Here we change it to dijkstras etc. according to given algo")
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
  #graph vertices are actually represented as numbers
  #like so: 0, 1, 2, ... V-1

  #pass the # of vertices and the graph to run prims algorithm
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
options = ["Original","Prim's", "Kruskal's", "Dijkstra's", "4", "5", "6", "7"]

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
