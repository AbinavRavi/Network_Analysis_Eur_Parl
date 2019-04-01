import numpy as np

import networkx as nx
import community
import network_utils as nu


def q_k_prime(network, sub, weighted=True):
    if len(sub.nodes()) == 1:
        return 0
    if weighted:
        n = sum([nx.degree(network, n, weight='weight') for n in network.nodes()])
        e_kk = sum([d['weight'] for u, v, d in sub.edges(data=True)])
        n_k = sum([nx.degree(sub, n, weight='weight') for n in sub.nodes()])
        d_k = sum([nx.degree(network, n, weight='weight') for n in sub.nodes()])
    else:
        n = len(network.nodes())
        e_kk = len(sub.edges())
        n_k = len(sub.nodes())
        d_k = sum([nx.degree(network, n) for n in sub.nodes()])

    p_k = 2 * e_kk / (n_k - 1)
    q_k = (d_k - 2 * e_kk) / (n - n_k)
    if p_k ==0:
        print(len(sub.nodes()))
        print(q_k)
    q_k_prime = q_k / p_k
    if q_k_prime > 1 or q_k_prime < 0:
        print(q_k_prime)

    return min(0, max(1, q_k_prime))


def reduce_weight(network, partition, weighted=True):
    values = [partition.get(n) for n in network.nodes()]
    nw = network.copy()
    parts = max(values) + 1
    for k in range(parts):
        part_k = [n for n in nw.nodes() if partition.get(n) == k]
        sub_k = nw.subgraph(part_k)

        q_k_p = q_k_prime(nw, sub_k, weighted)

        for u, v, d in sub_k.edges(data=True):
            d['weight'] *= q_k_p
    return nw


def reduce_edge(network, partition, weighted=True):
    values = [partition.get(n) for n in network.nodes()]
    nw = network.copy()
    parts = max(values) + 1
    for k in range(parts):
        part_k = [n for n in nw.nodes() if partition.get(n) == k]
        sub_k = nw.subgraph(part_k)

        q_k_p = q_k_prime(nw, sub_k, weighted)

        removeE = []
        for e in sub_k.edges():
            if np.random.random() > q_k_p:
                removeE.append(e)
        nw.remove_edges_from(removeE)

    return nw


def remove_edge(network, partition):
    nw = network.copy()
    removeE = [e for e in nw.edges() if partition[e[0]] == partition[e[1]]]
    nw.remove_edges_from(removeE)
    return nw


def hicode(network, weighted=True):
    networks = [network]
    partitions = [nu.get_partition(network)]
    level = layers(network)
    print('layers are',level)
    for i in range(level):
        # print(i)
        reduced_nw = reduce_weight(networks[-1], partitions[-1], weighted=True)
        # reduced_nw = remove_edge(networks[-1], partitions[-1])
        reduced_part = nu.get_partition(reduced_nw)
        networks.append(reduced_nw)
        partitions.append(reduced_part)
    return networks, partitions

def modularity(network):
    best_partition = community.best_partition(network)
    modularity = community.modularity(best_partition,network)
    return modularity

def layers(network,iter=5):
    networks = [network]
    partitions = [nu.get_partition(network)]
    modularity_before_refinement = modularity(network)
    mod = [modularity_before_refinement]
    nl=1
    for i in range(iter-1):
        mod.append(modularity(networks[-1]))
        # reduced_nw = reduce_weight(networks[-1], partitions[-1], weighted)
        reduced_nw = remove_edge(networks[-1], partitions[-1])
        reduced_part = nu.get_partition(reduced_nw)
        networks.append(reduced_nw)
        partitions.append(reduced_part)
        # print(mod)
        if (modularity_before_refinement < 0.1):
            return 1
        else:
            r_t = np.sum(mod)/(iter*mod[0]) 
            # print(r_t)
            if(r_t < 1):
                nl+=1
            else:
                nl-=1
    return(nl)

        



