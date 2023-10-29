import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import os.path as osp
import pickle as cp
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
import warnings
import multiprocessing as mp
import torch

from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx

# get the path
script_dir, script_name = osp.split(os.path.abspath(sys.argv[0]))
parent_dir = osp.dirname(script_dir)

# set paths of data, results and program parameters
DATA_BASE_PATH = parent_dir + '/data/'
RESULT_BASE_PATH = parent_dir + '/result/'


class GNNGraph(object):
    def __init__(self, g, label, node_tags, node_features):
        '''
            g: a networkx graph   子图
            label: an integer graph label  图标签：阳性样本为1，阴性样本为0
            node_tags: a list of integer node tags  子图中每个节点的标签，组成列表
            node_features: a numpy array of continuous node features  子图中每个节点的特征，组成数组
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features


        if len(g.edges()) != 0:
            x, y = list(zip(*g.edges()))
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
        
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(list(edge_features.values())[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in list(edge_features.items())}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)



def links2subgraphs(A, temp_train_data, temp_test_data ,temp_train_label,temp_test_label,node_feature): #h=2, max_nodes_per_hop=None,
    def helper(A, links, g_labels):
        '''
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information) #提取子图
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(GNNGraph(g, g_label, n_labels, n_features)) #把这些得到的子图都放进图列表里
        return g_list
        '''
        pbar = tqdm(zip(links[0],links[1],g_labels),unit = 'iteration')
        results_list = []
        for i,j ,z in pbar:
            ind = (i ,j)
            g_label = z
            results = subgraph_extraction_labeling(ind, A, g_label,node_feature,h=2, max_nodes_per_hop = 30) #max_nodes_per_hop=None
            if results is not None:
                results_list.append(results)
            else:
                pass

        graph_list = []
        for g,g_label, n_labels, n_features in results_list:
            g_list = GNNGraph(g, g_label, n_labels, n_features)
            graph_list.append(g_list)
        max_n_label['value'] = max(max([max(n_labels) for _,_, n_labels, _ in results_list]), max_n_label['value'])
        return graph_list
    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    print('Enclosing subgraph extraction begins...')

    train_graphs = helper(A,temp_train_data,temp_train_label)
    test_graphs = helper(A, temp_test_data, temp_test_label)
    return train_graphs, test_graphs, max_n_label['value']

def links2subgraphs1(A, temp_train_data, temp_test_data ,temp_train_label,temp_test_label,node_feature):
    def helper(A, links, g_labels):
        pbar = tqdm(zip(links[0],links[1],g_labels),unit = 'iteration')
        g_list = []
        results_list = []
        for i,j ,z in pbar:
            ind = (i ,j)
            g_label = z
            results = subgraph_extraction_labeling(ind, A, g_label, node_feature, h=2,max_nodes_per_hop=30)
            if results is not None:
                results_list.append(results)
            else:
                pass

        for g, g_label, n_labels, n_features in results_list:
            graph = Graph(g, g_label, n_labels, n_features)
            g_list.append(graph)
        max_n_label['value'] = max(max([max(n_labels) for _, _, n_labels, _ in results_list]), max_n_label['value'])
        return g_list
    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    print('Enclosing subgraph extraction begins...')
    train_graph = helper(A,temp_train_data,temp_train_label)
    test_graph = helper(A, temp_test_data, temp_test_label)
    return train_graph, test_graph,max_n_label['value']
def links2subgraphs2(A, temp_train_data, temp_test_data ,temp_train_label,temp_test_label,node_feature):
    def helper(A, links, g_labels):
        pbar = tqdm(zip(links[0],links[1],g_labels),unit = 'iteration')
        g_list = []
        results_list = []
        for i,j ,z in pbar:
            ind = (i ,j)
            g_label = z
            results= subgraph_extraction_labeling1(ind, A, g_label,node_feature,h=2, max_nodes_per_hop = 30)
            results_list.append(results)
        for g, g_label, n_labels, n_features in results_list:
            graph = Graph(g, g_label, n_labels, n_features)
            g_list.append(graph)
        max_n_label['value'] = max(max([max(n_labels) for _, _, n_labels, _ in results_list]), max_n_label['value'])
        return g_list
    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    print('Enclosing subgraph extraction begins...')
    train_graph = helper(A,temp_train_data,temp_train_label)
    test_graph = helper(A, temp_test_data, temp_test_label)
    return train_graph, test_graph,max_n_label['value']

class Graph(object):
    def __init__(self, g,label,node_tags, node_features):
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features
        if len(g.edges()) == 0:
            i, j = [], []
            self.num_edges = 0
        else:
            i, j = list(zip(*g.edges()))
            self.num_edges = len(i)
        self.edge_index = torch.LongTensor([i + j, j + i])


def nx_to_PyGGraph(batch_graphs,max_n_label):
    DATA = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        y = torch.Tensor([graph.label])
        node_tag = torch.zeros(graph.num_nodes, max_n_label+1)
        tags = graph.node_tags
        tags = torch.LongTensor(tags).view(-1,1)
        node_tag.scatter_(1, tags, 1)
        node_features = graph.node_features
        node_features = torch.FloatTensor(node_features)
        x = torch.cat([node_tag,node_features], 1)

        data = Data(x, graph.edge_index, y=y)
        DATA.append(data)
    return DATA

def subgraph_extraction_labeling1(ind, A,g_label,node_feature,h=2, max_nodes_per_hop=30):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)

    if node_feature is not None:
        node_feature = node_feature[nodes]
    #construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    #remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    return g, g_label,labels.tolist() , node_feature

def subgraph_extraction_labeling(ind, A,g_label,node_feature,h=2, max_nodes_per_hop=30):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    if len(nodes) == 2:
        print('target node {} has no hop neighbor.'.format(nodes))
    else:
        subgraph = A[nodes, :][:, nodes]
        # apply node-labeling
        labels = node_label(subgraph)
        if node_feature is not None:
             node_feature = node_feature[nodes]
        # construct nx graph
        g = nx.from_scipy_sparse_matrix(subgraph)
        #remove link between target nodes
        if not g.has_edge(0, 1):
           g.add_edge(0, 1)
        return g, g_label,labels.tolist() , node_feature


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0
    labels[labels<-1e6] = 0
    return labels


def to_linegraphs(batch_graphs, max_n_label):
    graphs = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        edges = graph.edge_pairs
        edge_feas = edge_fea(graph, max_n_label)/2
        edges, feas = to_undirect(edges, edge_feas)
        edges = torch.tensor(edges)
        data = Data(edge_index=edges, edge_attr=feas)
        data.num_nodes = graph.num_nodes
        data = LineGraph()(data)
        data['y'] = torch.tensor([graph.label])
        data.num_nodes = graph.num_edges
        graphs.append(data)
    return graphs

def edge_fea(graph, max_n_label):
    node_tag = torch.zeros(graph.num_nodes, max_n_label+1)
    tags = graph.node_tags
    tags = torch.LongTensor(tags).view(-1,1)
    node_tag.scatter_(1, tags, 1)
    node_features = graph.node_features
    node_features = torch.FloatTensor(node_features)
    x = torch.cat([node_tag,node_features], 1)
    return x

    
def to_undirect(edges, edge_fea):
    edges = np.reshape(edges, (-1,2 ))
    rp = np.array([edges[:,0], edges[:,1]], dtype=np.int64)
    fea_r = edge_fea[rp[0,:], :]
    fea_r = fea_r.repeat(2,1)
    fea_p = edge_fea[rp[1,:], :]
    fea_p = fea_p.repeat(2,1)
    fea_body = torch.cat([fea_r, fea_p], 1)
    pr = np.array([edges[:,1], edges[:,0]], dtype=np.int64)
    return np.concatenate([rp, pr], axis=1), fea_body


    
class Node:
    def __init__(self, name, serial_number, node_type):
        self.name = name
        self.interaction_list = []
        self.serial_number = serial_number
        self.embedded_vector = []
        self.attributes_vector = []
        self.node_type = node_type

class ncRNA:
    def __init__(self, ncRNA_name, serial_number, node_type):
        Node.__init__(self, ncRNA_name, serial_number, node_type)

class Protein:
    def __init__(self, protein_name, serial_number, node_type):
        Node.__init__(self, protein_name, serial_number, node_type)

class ncRNA_Protein_Interaction:
    def __init__(self, ncRNA, protein, y:int, key=None):
        self.ncRNA = ncRNA
        self.protein = protein
        self.y = y
        self.key = key

def printN(pred, target):
    TP = true_positive(pred, target)
    TN = true_negative(pred, target)
    FP = false_positive(pred, target)
    FN = false_negative(pred, target)
    print("TN:{},TP:{},FP:{},FN:{}".format(TN, TP, FP, FN))
    return TP,TN,FP,FN

def true_positive(pred, target):
    return ((pred == 1) & (target == 1)).sum().clone().detach().requires_grad_(False)


def true_negative(pred, target):
    return ((pred == 0) & (target == 0)).sum().clone().detach().requires_grad_(False)


def false_positive(pred, target):
    return ((pred == 1) & (target == 0)).sum().clone().detach().requires_grad_(False)


def false_negative(pred, target):
    return ((pred == 0) & (target == 1)).sum().clone().detach().requires_grad_(False)

def precision(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def sensitivity(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out

def specificity(pred, target):
    tn = true_negative(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)

    out = tn/(tn+fp)
    out[torch.isnan(out)] = 0

    return out


def MCC(pred,target):
    tp = true_positive(pred, target).to(torch.float)
    tn = true_negative(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)

    out = (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tn+fn)*(tp+fn)*(tn+fp))
    out[torch.isnan(out)] = 0

    return out

def accuracy(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    tn = true_negative(pred, target).to(torch.float)
    fp = false_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)
    out = (tp+tn)/(tp+tn+fn+fp)
    out[torch.isnan(out)] = 0

    return out


def FPR(pred, target):
    fp = false_positive(pred, target).to(torch.float)
    tn = true_negative(pred, target).to(torch.float)
    out = fp/(fp+tn)
    out[torch.isnan(out)] = 0
    return out


def TPR(pred, target):
    tp = true_positive(pred, target).to(torch.float)
    fn = false_negative(pred, target).to(torch.float)
    out = tp/(tp+fn)
    out[torch.isnan(out)] = 0
    return out

def AUC(label, prob):
    return roc_auc_score(label, prob)