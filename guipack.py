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
    plt.title("Here we change it to dijkstras etc. according to given algo")
    plt.show()
def filechoose_func():
        test = tkinter.Tk()
        test.withdraw()
        path = filedialog.askopenfilename(filetypes=(('text files', 'txt'),))
        f_read = open(path, "rt")
        print(f_read)
        #parser
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
        nodes = []
        edges = []
        z = []
        a = []
        b = []
        point = []
        e = []
        for j in range(num_nodes):  # extracted nodes
            a = [float(i) for i in y[0]]
            nodes.append(tuple(a[0:3]))
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

        for i in y:  # extract edges and convert in float
            for j in i:
                a.append(j)
            b.append(float(a[0]))
            for k in range(1, len(a)):
                b.append(float(a[k]))
                if k % 2 == 0:
                    edges.append(tuple(b[0:3]))
                    b.clear()
                    if k != (len(a) - 1):
                        b.append(float(a[0]))
            a.clear()

        for i in range(len(edges)):  # removes looped nodes
            if edges[i][0] == edges[i][1]:
                continue
            b.append(edges[i])

        edges.clear()
        for i in b:
            edges.append(i)
        b.clear()
        #parser
        #draw
        bandwidth = []
        for i in range(len(nodes)):
            point.append(nodes[i][1:3])

        for i in range(len(edges)):
            if edges[i][0] != edges[i][1]:
                a.append(int(edges[i][0]))
                a.append(int(edges[i][1]))
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
            plt.annotate(bandwidth[i] / 10000000,
                         ((((points[int(edges[i][0])][0]) * 0.75 + (points[int(edges[i][1])][0]) * 0.25),
                           (((points[int(edges[i][0])][1]) * 0.75 + (points[int(edges[i][1])][1]) * 0.25)))))
            while (1):
                if (start == edges[i + 1][0] and end == edges[i + 1][1]):
                    i += 1
                else:
                    i += 1
                    break
        #draw

def algoselected(event):
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

filechoose_button = tkinter.Button(app, image=filechoose_image,width=20,command=filechoose_func,borderwidth=0)
filechoose_button.grid(row=1,column=0,sticky="e",padx=20,pady=20)
filechoose_button.bind("<Enter>", on_enter)
filechoose_button.bind("<Leave>", on_leave)


algochoose_text = tkinter.Label(app, text='Choose Algorithm', font=('bold', 12),bg='blue',fg='white')
algochoose_text.grid(row=0, column=1,sticky='s')
options = ["Select Algorithm","1", "2", "3", "4", "5", "6", "7"]

algochoose = StringVar()
algochoose.set(options[0])
drop = tkinter.OptionMenu(app, algochoose, *options,command=algoselected)
drop.config(bg='lightgreen')
drop.grid(row=1, column=1,sticky='n',pady=20)


endprog = tkinter.Button(app, text = 'End Program', command = exiter,bg='white',fg='black')
endprog.grid(row = 0, column = 2,sticky='sw')
endprog.bind("<Enter>", on_enter_endprog)
endprog.bind("<Leave>", on_leave_endprog)

draw_button = tkinter.Button(app, text ="Draw",width=12,command=draw,bg='white',fg='black')
draw_button.grid(row=1,column=2,sticky='nw',pady=20)
draw_button.bind("<Enter>", on_enter)
draw_button.bind("<Leave>", on_leave)
app= mainloop()
