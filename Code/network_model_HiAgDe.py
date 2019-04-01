import numpy as np
import scipy as sp
import pandas as pd
import ast

import gensim
from gensim.corpora import Dictionary

import networkx as nx
import network_utils as nu

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -----------------------------------------------------------------------------------------------------------------------
## Loading data
topicDF = pd.read_csv('../data/topicData.csv')
topicDF['date'] = pd.to_datetime(topicDF['date'])
textDF = pd.read_csv('../data/sessionData.csv')
textDF['date'] = pd.to_datetime(textDF['date'])

# topicDF_part = topicDF[(topicDF.date < '2001-07-01') & (topicDF.date >= '2000-07-01')]
topicDF_part = topicDF[topicDF.date == '2001-09-01']
# textDF = textDF[(textDF.date < '2001-07-01') & (textDF.date >= '2000-07-01')]
textDF = textDF[textDF.date == '2001-09-01']

ldamodel = gensim.models.ldamodel.LdaModel.load('notebooks/model.gensim')
dictionary = Dictionary.load('notebooks/dictionary.gensim')

# -----------------------------------------------------------------------------------------------------------------------
## Building network
network = nu.build_network(topicDF_part, 15, exclude=[1, 9])
bottom_nodes = [n for n in network.nodes() if n not in range(15)]
network = nu.fold_network(network, bottom_nodes, mode='multi')
network = nu.normalize_edgeweight(network)
part, nw = nu.get_partition(network)
values = [part.get(n) for n in nw.nodes()]

# -----------------------------------------------------------------------------------------------------------------------
## Analyzing network
count = [{n: 0 for n in nw.nodes()} for i in range(15)]
rownr = textDF.shape[0]
word2id = dict((ldamodel.id2word[id],id) for id in ldamodel.id2word)
term_topic = ldamodel.get_topics()
for i in range(rownr):
    mep = textDF['name'].iloc[i]
    text = ast.literal_eval(textDF['text'].iloc[i])
    for j in range(15):
        if mep in count[j].keys():
            for word in text:
                count[j][mep] += term_topic[j][word2id[word]]

top_list = []
for i in range(max(values) + 1):
    sub = network.subgraph([n for n in network.nodes() if part.get(n) == i]).copy()
    sub = nu.normalize_edgeweight(sub)
    top, thresh = nu.get_top_topic(sub)
    top_list.append(top)
print(top_list)

# -----------------------------------------------------------------------------------------------------------------------
## Drawing network
plt.style.use('classic')

removeE = [e for e in nw.edges() if part[e[0]] != part[e[1]]]
nw.remove_edges_from(removeE)

pos = nx.spring_layout(nw, iterations=15, weight='weight')
sizes = [50 * nu.node_weight(nw, node) for node in nw.nodes()]
weights = [nw[u][v]['weight'] for u, v, in nw.edges()]

nc = nx.draw_networkx_nodes(nw, pos, with_labels=False, node_color=values, node_size=sizes, alpha=0.4,
                            cmap=cm.gist_rainbow)
nx.draw_networkx_edges(nw, pos, width=weights)

plt.axis('off')
plt.colorbar(nc)

for i in top_list:
    plt.figure()
    colors = [count[i][n] for n in nw.nodes()]

    nc = nx.draw_networkx_nodes(nw, pos, with_labels=False, node_color=colors, node_size=sizes, alpha=0.4,
                                cmap=cm.YlOrRd)
    nx.draw_networkx_edges(nw, pos, width=weights)

    plt.axis('off')
    plt.colorbar(nc)
plt.show()
