import numpy as np
import scipy as sp
import pandas as pd
import ast
import itertools
from itertools import product
from collections import Counter

import networkx as nx
import network_utils as nu
import hicode as hc

import matplotlib.pyplot as plt
import matplotlib.cm as cm


plt.style.use('classic')

# -----------------------------------------------------------------------------------------------------------------------
## Loading data
topicDF = pd.read_csv('../Topics/topicsData350.csv')
topicDF['date'] = pd.to_datetime(topicDF['date'])
# topicDF_part = topicDF[(topicDF.date < '2001-07-01') & (topicDF.date >= '2000-07-01')]
# topicDF_part = topicDF[topicDF.date == '2000-07-01']
sit = 0
count = Counter([])
for i in range(58):
    year = 1999 + (i + 6) // 12
    month = (i + 6) % 12 + 1
    date = '{:4d}-{:02d}-01'.format(year, month)
    year = 1999 + (i + 9) // 12
    month = (i + 9) % 12 + 1
    date2 = '{:4d}-{:02d}-01'.format(year, month)
    topicDF_part = topicDF[(topicDF.date < date2) & (topicDF.date >= date)]

    if topicDF_part.shape[0] == 0:
        continue
    else:
        sit += 1
    f = open('../data/outliers.txt', 'a')
    f.write('{:s}\n'.format(date))
    print(date)

# -----------------------------------------------------------------------------------------------------------------------
## Building network
    network = nu.build_network(topicDF_part, 350, exclude=[])
    #print(len(network.nodes()))
    bottom_nodes = [n for n in network.nodes() if n not in range(350)]
    network = nu.fold_network(network, bottom_nodes, mode='single')
    network = nu.normalize_edgeweight(network)

# -----------------------------------------------------------------------------------------------------------------------
## Analyzing network

    networks, partitions = hc.hicode(network, True)

    candidates = [(u, v) for u, v in product(network.nodes(), network.nodes()) if
              u != v and partitions[0][u] != partitions[0][v]]
    for i in range(1,len(partitions)):
        candidates = [(u,v) for u, v in candidates if partitions[i][u] == partitions[i][v]]
    candidates = [(u,v) for u,v in candidates]
    # candidates.sort()
    # candidates = list(k for k,_ in itertools.groupby(candidates))
    # print(candidates)
    # candidates = [tuple(c) for c in candidates ]
    count+=Counter(candidates)

count = dict(count)
count = sorted(count.items(), key=lambda kv: kv[1], reverse=True)
with open('../Results_Hicode/first_session_redweight.txt', 'w') as f:
    f.write('Total sittings: {:d}\n\n'.format(int(sit)))
    for k, v in count:
        f.write('{:s}: {:d}, {:f}\n'.format(str(k), int(v), v / sit))
# -----------------------------------------------------------------------------------------------------------------------
## Drawing network
# for i in range(len(networks)):
#     plt.figure()
#     values = [partitions[0].get(n) for n in networks[i].nodes()]
#     removeE = [e for e in networks[i].edges() if partitions[i][e[0]] != partitions[i][e[1]]]
#     networks[i].remove_edges_from(removeE)

#     pos = nx.spring_layout(networks[i], iterations=15, weight='weight')
#     sizes = [50 * nu.node_weight(networks[i], node) for node in networks[i].nodes()]
#     weights = [networks[i][u][v]['weight'] for u, v, in networks[i].edges()]

#     nc = nx.draw_networkx_nodes(networks[i], pos, with_labels=False, node_color=values, node_size=sizes, alpha=0.4,
#                                 cmap=cm.gist_rainbow)
#     nx.draw_networkx_edges(networks[i], pos, width=weights)

#     plt.axis('off')
#     plt.colorbar(nc)
# plt.show()
