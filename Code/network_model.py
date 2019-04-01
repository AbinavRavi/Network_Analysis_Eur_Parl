import numpy as np
import scipy as sp
import pandas as pd
import string
from collections import Counter
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast

def weight_function(G,u,v):
    w = 0
    for node in G[u].keys():
        if node in G[v].keys():
            w = w + G[u][node]['weight'] + G[v][node]['weight']
    return w/200

def node_weight(G,u):
    w = 0
    for v in G[u].keys():
        w = w + G[u][v]['weight']
    return w

def get_topics(topics, mode='p', top=0.5):
    t = sorted(topics, key=lambda x:x[1], reverse=True)
    t2 = []
    s = 0
    i = 0
    if mode == 'p':
        while s < top:
            t2.append(t[i])
            s += t[i][1]
            i += 1
    elif mode == 'n':
        while i < top:
            t2.append(t[i])
            i += 1
    return t2


## Loading data

topicDF = pd.read_csv('../data/topicData.csv')
rownr = topicDF.shape[0]


## Building network

network = nx.Graph()
topics = set()
for i in range(rownr):
    if not i%1000:
        print(i)
    mep = topicDF['name'].iloc[i]
    topic = get_topics(ast.literal_eval(topicDF['topic'].iloc[i]))
    for t in topic:
        if t[0] not in [2,3,8,11,14] and t[1] > 0:
            topics.add(t[0])
            edge = (mep,t[0])
            if edge in network.edges():
                network[mep][t[0]]['weight'] += t[1]
            else:
                network.add_edge(mep, t[0], weight=t[1])

bp = dict((n,n in topics) for n in network.nodes())
nx.set_node_attributes(network, bp, 'bipartite')
top_nodes = [n for n, d in network.nodes(data=True) if d['bipartite']==1]
bottom_nodes = [n for n, d in network.nodes(data=True) if d['bipartite']==0]
#network = bipartite.generic_weighted_projected_graph(network,bottom_nodes,weight_function=weight_function)

w = [network[e[0]][e[1]]['weight'] for e in network.edges()]
thresh = min(w) + (max(w) - min(w)) * 0.5
print(thresh)

removeE = [e for e in network.edges() if network[e[0]][e[1]]['weight'] < thresh]
network.remove_edges_from(removeE)

removeN = [node for node in network.nodes() if dict(network.degree())[node] == 0]
network.remove_nodes_from(removeN)

print(len(network.nodes()))

## Analyzing network

W = nx.to_numpy_matrix(network, nodelist=network.nodes())
D = np.diag(np.sum(W,axis=1).A1)
L = D - W
D_sqrt = np.diag(np.sqrt(1/np.diag(D)))
L_symm = np.dot(D_sqrt,np.dot(L,D_sqrt))

k = 2
_, v = sp.linalg.eigh(L_symm,eigvals=(1,k))


## Drawing network


if k == 2:
    plt.subplot(1,2,2)
    plt.plot(v[:,0],v[:,1],'.r')


    plt.subplot(1,2,1)
    pos=nx.spring_layout(network,iterations=5, weight='weight')
    #pos=nx.shell_layout(network,nlist=[top_nodes,bottom_nodes])
    
    colors = ['b' if node in topics else 'g' if dict(network.degree())[node] == 1 else'r' for node in network.nodes()]
    sizes = [500 if node in topics else 10 * node_weight(network, node) for node in network.nodes()]
    weights = [network[u][v]['weight'] for u,v, in network.edges()]
    labels = {}
    for i in topics:
        if i in network.nodes():
            labels[i] = i

    nx.draw_networkx_nodes(network, pos, with_labels=False, node_color=colors, node_size=sizes, alpha=0.4)
    nx.draw_networkx_edges(network, pos, width=weights)
    if len(labels) > 0:
        nx.draw_networkx_labels(network,pos,labels,font_color='w')
    plt.axis('off')

elif k > 2:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(v[:,0],v[:,1],v[:,2],'.r')
plt.show()

