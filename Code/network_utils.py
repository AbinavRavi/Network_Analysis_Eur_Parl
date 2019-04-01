import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from scipy.stats import multinomial
import pandas as pd

import ast
import numbers
from collections import Counter
import operator

import networkx as nx
from networkx.algorithms import bipartite
import community

import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('classic')


# ---------------------------------------------------------------------------------------------------------------------
## Network building utilities

def single_weight_function(G, u, v):
    """
    Function used to fold the network to a singleweight representation
    :param G: networkx.Graph
            Graph containing u and v
    :param u: node
            First node
    :param v: node
            Second node
    :return:
        w: double
         weight of the edge (u,v) in the folded graph
    """
    w = 0
    for node in G[u].keys():
        if node in G[v].keys():
            w = w + G[u][node]['weight'] * G[v][node]['weight']
    return w


def multi_weight_function(G, u, v):
    """
    Function used to fold the network to a multiweight representation
    :param G: networkx.Graph
            Graph containing u and v
    :param u: node
            First node
    :param v: node
            Second node
    :return:
        w: dictionary
         weight of the edge (u,v) in the folded graph
    """
    w = {}
    for node in G[u].keys():
        if node in G[v].keys():
            w[node] = G[u][node]['weight'] * G[v][node]['weight']
    return w


def node_weight(G, u):
    """
    Computes the weighted degree of a node
    :param G: networkx.Graph
            Graph containing the node u
    :param u: node
            Node of which the degree will be computed
    :return:
        w: double
         Degree of u
    """
    w = 0
    for v in G[u].keys():
        w = w + G[u][v]['weight']
    return w


def get_topics(topics, mode='p', top=0.5):
    """
    Returns the top topics from a list of topics with corresponding probabilities
    :param topics: list of tuples (topic, double)
            List of the topics with corresponding probabilities
    :param mode: str, optional
            If 'p' top percentage of topics will be considered
            If 'n' top number of topics will be considered
            Default to 'p'
    :param top: double, optional
            If mode = 'p' the top topics having their probability sum > top will be chosen
            If mode = 'n' the top top many topics will be chosen
            Default to 0.5
    :return:
        t: list
         List containing the top topics
    """
    t = sorted(topics, key=lambda x: x[1], reverse=True)
    t2 = []
    s = 0
    i = 0
    if mode == 'p':
        while s < top and i < len(t):
            t2.append(t[i])
            s += t[i][1]
            i += 1
    elif mode == 'n':
        while i < top and i < len(t):
            t2.append(t[i])
            i += 1
    return t2


# ---------------------------------------------------------------------------------------------------------------------
##

def build_network(df, nr_topics, key='name', topic_mode='p', topic_top=1, exclude=[], topic_thresh=0):
    """
    Builds a two-mode network from a pandas Dataframe.

    :param df: pandas.DataFrame
            Dataframe with columns key and 'topic'
    :param nr_topics: int
            Number of topics
    :param key: str, optional
            Column of the dataframe that is used as one part of the nodes
            Default to 'name'
    :param topic_mode: str, optional
            If 'p' top percentage of topics will be considered
            If 'n' top number of topics will be considered
            Default to 'p'
    :param topic_top: double, optional
            If topic_mode = 'p' the top topics having their probability sum > topic_top will be chosen
            If topic_mode = 'n' the top topic_top many topics will be chosen
            Default to 1
    :param exclude: list, optional
            List of topics to exclude when building the network
            Default to []
    :param topic_thresh: double, optional
            Minimum probability for topic to be included
            Default to 0
    :return:
        network: networkx.Graph
    """
    network = nx.Graph(nr_topics=nr_topics)
    rownr = df.shape[0]

    for i in range(rownr):
        k = df[key].iloc[i]
        topics = get_topics(ast.literal_eval(df['topic'].iloc[i]), mode=topic_mode, top=topic_top)
        if k not in network.nodes():
            network.add_node(k, topics=np.zeros(nr_topics))
        for t in topics:
            if t[0] not in exclude and t[1] > topic_thresh:
                edge = (k, t[0])
                if edge not in network.edges():
                    network.add_edge(k, t[0], weight=0)
                network[k][t[0]]['weight'] += t[1]
                network.nodes[k]['topics'][t[0]] += t[1]
    bp = dict((n, n in range(nr_topics)) for n in network.nodes())
    nx.set_node_attributes(network, bp, 'bipartite')
    return network


def fold_network(network, nodes, mode='multi'):
    """
    Folds the network from a two-mode representation to a one-mode representation.
    :param network: networkx.Graph
            Bipartite graph to be folded
    :param nodes: list of nodes
            The node set to keep
    :param mode: str, optional
            'multi' for getting an edge weight dictionary
            'single' for getting a single weight per edge
            Default to 'multi'
    :return:
        nw: networkx.Graph
         The folded network
    """
    if mode == 'multi':
        return bipartite.generic_weighted_projected_graph(network, nodes, weight_function=multi_weight_function)
    elif mode == 'single':
        return bipartite.generic_weighted_projected_graph(network, nodes, weight_function=single_weight_function)
    else:
        return None


def normalize_edgeweight_old(network):
    """
    Deprecated
    """
    e = list(network.edges())[0]
    w = network[e[0]][e[1]]['weight']

    if isinstance(w, numbers.Number):
        w_max = max(network[e[0]][e[1]]['weight'] for e in network.edges())
        for u, v, d in network.edges(data=True):
            d['weight'] /= w_max
    elif type(w) is dict:
        w_max = {}
        for e in network.edges():
            w_max = {k: max(i for i in (w_max.get(k), network[e[0]][e[1]]['weight'].get(k)) if i) for k in
                     w_max.keys() | network[e[0]][e[1]]['weight']}
        for u, v, d in network.edges(data=True):
            for k in d['weight'].keys():
                d['weight'][k] /= w_max[k]
    else:
        return None
    return network


def normalize_edgeweight(network):
    """
    Normalizes the edge weights to be between 0 and 1.
    :param network: networkx.Graph
            Graph with unnormalized weight attributes on the edges
    :return:
        network: networkx.Graph
         Graph with normalized weight attributes on the edges
    """
    if len(network.edges()) == 0:
        return network
    e = list(network.edges())[0]
    w = network[e[0]][e[1]]['weight']

    if isinstance(w, numbers.Number):
        w_max = max(network[e[0]][e[1]]['weight'] for e in network.edges())
        for u, v, d in network.edges(data=True):
            d['weight'] /= w_max
    elif type(w) is dict:
        w_max = max(max(network[e[0]][e[1]]['weight'].values()) for e in network.edges())
        for u, v, d in network.edges(data=True):
            for k in d['weight'].keys():
                d['weight'][k] /= w_max
    else:
        return None
    return network


def to_attributed_network(network, mode='continuous'):
    nw = network.copy()
    if mode == 'continuous':
        nr_topics = nw.graph['nr_topics']
        for u, v, d in nw.edges(data=True):
            d['topics'] = np.zeros(nr_topics)
            weights = d['weight']
            d['topics'][list(weights.keys())] = list(weights.values())
            d['weight'] = sum(weights.values())
    elif mode == 'discrete':
        for u, v, d in nw.edges(data=True):
            weights = d['weight']
            d['topics'] = get_topics(list(weights.items(), mode='n', top=1)[0][0])
            d['weight'] = sum(weights.values())
    else:
        return None
    return nw


def sum_weights(network, topic=None, thresh=0.5, remove_unconnected=False):
    """
    Transforms a network with dict edgeweight to a network with float edgeweight
    :param network: networkx.Graph
            Graph with dict edge attribute 'weight'
    :param topic: optional
            If topic is not None only edges with this topic having edge weight > thresh will be kept
            Default to None
    :param thresh: double, optional
            Threshold for keeping an edge
            Default to 0.5
    :param remove_unconnected: boolean, optional
            If True nodes with degree 0 will be removed after truncating edges
            Default to False
    :return:
        nw: networkx.Graph
         Graph with single double value as edge weight
    """
    nw = nx.Graph()
    for u, v, d in network.edges(data=True):
        if (topic is None) or (topic in d['weight'].keys() and d['weight'][topic] > thresh):
            nw.add_edge(u, v, weight=sum(d['weight'].values()))
    if not remove_unconnected:
        for n in network.nodes():
            if n not in nw.nodes():
                nw.add_node(n)
    return nw


def get_partition(network, topic=None, thresh=0.5, remove_unconnected=False):
    """
    Partitions the network using the Louvain algorithm
    :param network: networkx.Graph
            Network to be partitioned
    :param topic: optional
            If topic is not None only edges with this topic having edge weight > thresh will be kept
            Default to None
    :param thresh: double, optional
            Threshold for keeping an edge
            Default to 0.5
    :param remove_unconnected: boolean, optional
            If True nodes with degree 0 will be removed after truncating edges
            Default to False
    :return:
        partition: dictionary
         Partition with communities numbered 0 to number of communities
        nw: networkx.Graph, optional
         If input network was multiweight, returns singleweight network
    """
    if(len(network.edges())==0):
        return community.best_partition(network)
    e = list(network.edges())[0]
    w = network[e[0]][e[1]]['weight']

    if isinstance(w, numbers.Number):
        return community.best_partition(network)
    elif type(w) is dict:
        nw = sum_weights(network, topic, thresh, remove_unconnected)
        return [community.best_partition(nw), nw]


def get_top_topic(network):
    """
    Calculates the topic with highest edge weight sum
    :param network: networkx.Graph
            Graph with dict as edge attribute 'weight'
    :return:
        top:
         Topic with highest edge weight sum
        thresh: double
         Weight sum of top topic / number of edges
    """
    topic_dict = Counter({})
    for u, v, d in network.edges(data=True):
        topic_dict += Counter(d['weight'])
    topic_dict = dict(topic_dict)
    top = max(topic_dict.items(), key=operator.itemgetter(1))[0]
    thresh = max(topic_dict.values()) / len(network.edges())
    return top, thresh


def get_majority_topics(network, q=0.5):
    topic_dict = Counter({})
    for u, v, d in network.edges(data=True):
        topic_dict += Counter(d['weight'])
    topic_list = sorted(list(topic_dict.items()), key=lambda x: x[1], reverse=True)
    major_topics = []
    val_sum = sum(topic_dict.values())
    quant = 0
    edge_len = len(network.edges())
    for t in topic_list:
        if quant < q:
            major_topics.append((t[0], t[1] / edge_len, t[1] / val_sum))
            quant += t[1] / val_sum
        else:
            break
    return major_topics


def get_outliers(network, mode='percentile', type='edge'):
    if len(network.edges()) == 0:
        return []
    nw = to_attributed_network(network)
    if type == 'edge':
        topics = np.array([d['topics'] for u, v, d in nw.edges(data=True)])
    elif type == 'node':
        topics = np.array([d['topics'] for n, d in nw.nodes(data=True)])
    else:
        return None
    if mode == 'percentile':
        upper = np.percentile(topics, 75, axis=0)
        lower = np.percentile(topics, 25, axis=0)
        iqr = upper - lower
        upper_out = upper + 1.5 * iqr
        lower_out = lower - 1.5 * iqr
        '''
        plt.boxplot(topics)
        plt.show()
        '''
    elif mode == 'std':
        mean = np.mean(topics, axis=0)
        std = np.std(topics, axis=0)
        upper_out = mean + 2 * std
        lower_out = mean - std
        '''
        outlier_bool = np.any(topics < lower_out, axis=1)
        if 'Rolf Linkohr' in nw.nodes() or True:
            n, m = topics.shape
            n = np.sum(outlier_bool)
            for j in range(n):
                plt.figure()
                plt.plot(list(range(m)), topics[outlier_bool, :][j,:], marker='o', color='black')
                plt.errorbar(list(range(m)), mean, std, linestyle='None', marker='.', capsize=5, ecolor='r')
                break
            plt.show()
        '''
    else:
        return None
    # outlier_bool = np.any((topics > upper_out) | (topics < lower_out), axis=1)
    outlier_bool = np.any(topics < lower_out, axis=1)
    if type == 'edge':
        outlier = [e for i, e in enumerate(nw.edges()) if outlier_bool[i]]
    elif type == 'node':
        outlier = [n for i, n in enumerate(nw.nodes()) if outlier_bool[i]]
    else:
        return None
    return outlier


# ---------------------------------------------------------------------------------------------------------------------
## HCOutlier Detection

def hco_detection(network, K, Z_start=None, l=[1, 1], rho=5000, eps=1 + 1e-5, max_iter=2000, edges=False):
    E = len(network.edges())
    N = len(network.nodes())
    nr_topics = network.graph['nr_topics']

    W_vv = nx.to_numpy_matrix(network, nodelist=network.nodes())
    W_ve = np.zeros((N, E))
    if edges:
        i = 0
        for n in network.nodes():
            j = 0
            neigh = network.neighbors(n)
            w = 1 / len(list(neigh))
            for u, v in network.edges():
                if u == n and v in neigh:
                    W_ve[i, j] = w
                j += 1
            i += 1

    if Z_start is not None:
        Z_v = Z_start[0]
        Z_e = Z_start[1]
    else:
        Z_v = np.random.randint(K + 1, size=N)
        Z_e = np.random.randint(K + 1, size=E)
    mu = np.zeros((K + 1, nr_topics))
    sigma = np.stack([np.eye(nr_topics)] * (K + 1))
    S_v = np.zeros((N, nr_topics))
    S_e = np.zeros((E, nr_topics))
    i = 0
    for n, d in network.nodes(data=True):
        S_v[i, :] = d['topics']
        i += 1
    # S_v /= np.sum(S_v, axis=1)[:, None]
    if edges:
        i = 0
        for u, v, d in network.edges(data=True):
            S_e[i, :] = d['topics']
            i += 1
        S_e /= np.sum(S_e, axis=1)[:, None]
    S = [S_v, S_e]

    i = 0
    Z_old_v = Z_v + np.ones_like(Z_v)
    Z = [Z_v, Z_e]
    while i < max_iter and np.linalg.norm(Z_old_v - Z[0]) > eps:
        # print(i)
        Z_old_v = Z[0]
        i += 1
        mu, sigma = update_parameters(Z, mu, sigma, S, rho=rho, edges=edges)
        Z = infer_hidden_labels(Z, mu, sigma, [W_vv, W_ve], S, l=l, edges=edges)
    print(i)
    return {n: Z[0][i] for n, i in zip(network.nodes(), range(N))}


def update_parameters(Z, mu_old, sigma_old, S, rho=5000, edges=False):  # 1000 < rho < 10000
    Z_v, Z_e = Z
    S_v, S_e = S
    K, _ = mu_old.shape
    N, nr_topics = S_v.shape
    E, _ = S_e.shape
    mu = np.zeros((K, nr_topics))
    sigma = np.zeros((K, nr_topics, nr_topics))
    p_v = np.zeros((K, N))
    p_e = np.zeros((K, E))

    for k in range(K):  # TODO split p_k?
        if edges:
            p_k = (np.sum(Z_v == k) + np.sum(Z_e == k)) / (N + E)
        else:
            p_k = np.sum(Z_v == k) / N
        if k == 0:
            p_v[k, :] += rho
            if edges:
                p_e[k, :] += rho
        else:
            for b in range(N):
                p_v[k, b] = multivariate_normal.pdf(S_v[b, :], mean=mu_old[k, :], cov=sigma_old[k, :, :],
                                                    allow_singular=True) * p_k
            if edges:
                for b in range(E):
                    p_e[k, b] = multivariate_normal.pdf(S_e[b, :], mean=mu_old[k, :], cov=sigma_old[k, :, :],
                                                        allow_singular=True) * p_k

    norm_v = np.sum(p_v, axis=0)[None, :]
    norm_v[norm_v == 0] = 1
    p_v /= norm_v
    if edges:
        norm_e = np.sum(p_e, axis=0)[None, :]
        norm_e[norm_e == 0] = 1
        p_e /= norm_e
    norm = np.sum(p_v, axis=1)
    if edges:
        norm += np.sum(p_e, axis=1)
    norm[norm == 0] = 1

    for k in range(K):
        for b in range(N):
            mu[k, :] += p_v[k, b] * S_v[b, :]
        if edges:
            for b in range(E):
                mu[k, :] += p_e[k, b] * S_e[b, :]
        mu[k, :] /= norm[k]

        S_v_norm = S_v - mu[k, :]
        for b in range(N):
            sigma[k, :, :] += p_v[k, b] * np.outer(S_v_norm[b, :], S_v_norm[b, :])
        if edges:
            S_e_norm = S_e - mu[k, :]
            for b in range(E):
                sigma[k, :, :] += p_e[k, b] * np.outer(S_e_norm[b, :], S_e_norm[b, :])
    sigma /= norm[:, None, None]
    return mu, sigma


def infer_hidden_labels(Z_old, mu, sigma, W, S, l=[1.2, 1], edges=False):
    K, nr_topics = mu.shape
    l1, l2 = l
    W_vv, W_ve = W
    Z_old_v, Z_old_e = Z_old
    S_v, S_e = S
    N = len(Z_old_v)
    E = len(Z_old_e)

    U_v = np.zeros((N, K))
    U_e = np.zeros((E, K))

    for b in range(N):
        max = []
        for k in range(K):
            if k != 0:
                U_v[b, k] -= l1 * np.sum(np.multiply(W_vv[b, :], (Z_old_v == k)))
                if edges:
                    U_v[b, k] -= l2 * np.sum(np.multiply(W_ve[b, :], (Z_old_e == k)))
            p = 0
            p = multivariate_normal.pdf(S_v[b, :], mean=mu[k, :], cov=sigma[k, :, :], allow_singular=True)
            if p != 0:
                U_v[b, k] -= np.log(p)
            else:
                max.append(k)
        m = np.max(U_v[b, :]) + 1
        for k in max:
            U_v[b, k] = m
    Z_v = np.argmin(U_v, axis=1)

    if edges:
        for b in range(E):
            max = []
            for k in range(K):
                U_e[b, k] -= l2 * np.sum(np.multiply(W_ve[:, b], (Z_old_v == k)))
                p = 0
                p = multivariate_normal.pdf(S_e[b, :], mean=mu[k, :], cov=sigma[k, :, :], allow_singular=True)
                if p != 0:
                    U_e[b, k] -= np.log(p)
                else:
                    max.append(k)
            m = np.max(U_e[b, :]) + 1
            for k in max:
                U_e[b, k] = m
        Z_e = np.argmin(U_e, axis=1)
    else:
        Z_e = Z_old_e

    return [Z_v, Z_e]
