import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import pandas as pd
import torch_geometric
import warnings
from tqdm import tqdm
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
warnings.simplefilter("ignore")
import dgl.data
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from dgl.nn import GraphConv, TAGConv, NNConv, EdgeWeightNorm, RelGraphConv, GATv2Conv, EGATConv, EdgeConv
import itertools


from homophily_analysis_tools.homophily_metrics import class_homophily, edge_homophily, node_homophily
from homophily_analysis_tools.gnn_models import train_GCN_g
from homophily_analysis_tools.generate_synthetic_graphs import gen_synthetic_rewired_graph,gen_synthetic_SBM_graph_list

def shotest_paths(g,G=None,with_tqdm=True,n_max=None):
  """
  Compute the shortest paths between all nodes in a graph and return them as a numpy array of shape (n_edges,3) 
  where the first column is the source node, the second column is the target node and the third column 
  is the length of the shortest path between the source and target node.


  Args:
      g: dgl graph
      G: networkx graph (optional)
      with_tqdm: whether to use tqdm to show progress
      n_max: maximum number of nodes to compute shortest paths for (optional) 

  Returns:
      spl: numpy array of shortest paths between all nodes in graph

  
  """
  transform = dgl.RemoveSelfLoop()
  g = transform(g)
  if not n_max:
    n_max = g.number_of_nodes()

  if not G:
    G = nx.Graph(dgl.to_networkx(g))
  s_G = nx.shortest_path_length(G)
  y = []
  if with_tqdm:
    with tqdm(total=int(n_max)) as pbar:
      for i,x in enumerate(s_G):
        out_nodes = list(x[1].keys())
        in_node = [x[0]]*len(out_nodes)
        lengths = list(x[1].values())
        y.append(np.array(list(zip(in_node,out_nodes,lengths)
        )))
        if i >n_max:
          break
        pbar.update(1)
  else:
    for i,x in enumerate(s_G):
      out_nodes = list(x[1].keys())
      in_node = [x[0]]*len(out_nodes)
      lengths = list(x[1].values())
      y.append(np.array(list(zip(in_node,out_nodes,lengths)
      )))
      if i >n_max:
        break

  spl = np.concatenate(y)

  
  return spl


def shotest_paths_dist(g,n_max=500,nodes=None):
  """
  Compute the distribution of shortest paths between all nodes in a graph. Returns the shortest path lengths 
  and the proportion of shortest paths of that length. 

  Args:
      g: dgl graph
      n_max: maximum number of nodes to compute shortest paths for (optional)
      nodes: nodes to compute shortest paths for (optional)

  Returns:
      lengths: shortest path lengths
      spl_dist: proportion of shortest paths of each length

  """
  if nodes is None:
    nodes = g.nodes().numpy()
  
  spl1 = spl_between_nodes(g,nodes,n_max=n_max)
  lengths = np.unique(spl1[:,2])
  spl_dist = np.zeros(lengths.shape[0])
  for i,l in enumerate(lengths):
    spl_dist[i] = (spl1[:,2]==l).sum()

  
  return lengths,spl_dist/sum(spl_dist)


def spl_between_nodes(g,nodes,n_max=500):
  spl1 = shotest_paths(g,with_tqdm=False,n_max = n_max)
  spl1 = spl1[np.isin(spl1[:,0],nodes)]
  spl1 = spl1[np.isin(spl1[:,1],nodes)]
  return spl1


def mean_spl(g,n_max=500,nodes=None):
  """
  Compute the mean shortest path length between all nodes in a graph. Optionally only compute the mean shortest path length
  between a subset of nodes. 

  Args:
      g: dgl graph
      n_max: maximum number of nodes to compute shortest paths for (optional)
      nodes: nodes to compute shortest paths for (optional)

  Returns:
      mean shortest path length

  """

  if nodes is None:
    nodes = g.nodes().numpy()
  length,freq = shotest_paths_dist(g,n_max=n_max,nodes=nodes)
  return (length*freq).sum()


def intra_class_mean_spl(g):
  """
  Compute the mean shortest path length between nodes of the same class in a graph. 

  Args:
      g: dgl graph

  Returns:
      mean shortest path length between nodes of the same class
  
  """
  classes = g.ndata['label'].unique()
  s = 0
  for c in classes:
    s += mean_spl(g,nodes=g.nodes()[g.ndata['label']==c])
  return s/len(classes)


def calculate_geodesics_homophily(g_list):
  """
  Given a list of pre-generated graphs, calculate the mean shortest path length between 
  nodes of the same class. Optionally calculate the mean shortest path length between all nodes in the graph.
  Also calculate the homohiply of the graphs produced.

  Args:
      g_list: list of dgl graphs
      do_full_spl: whether to calculate the mean shortest path length between all nodes in the graph

  Returns:
      mean_spl_list: mean shortest path length between all nodes in the graph
      mean_intra_spl_list: mean shortest path length between nodes of the same class
  """
  mean_spl_list = []
  mean_intra_spl_list = []

  for g in tqdm(g_list):
    mean_intra_spl_g = intra_class_mean_spl(g)
    mean_spl_g = mean_spl(g,n_max=500)

    mean_spl_list.append(mean_spl_g)
    mean_intra_spl_list.append(mean_intra_spl_g)

  return mean_spl_list,mean_intra_spl_list


def test_gnn_vs_homophily(g_list):
  """
  Test the performance of a GNN on a graph classification task.

  Args:
      g_list: list of dgl graphs

  Returns:
      acc_list: list of GNN accuracy values

  """
  acc_list = []
  
  for g in tqdm(g_list):
    acc = train_GCN_g(g,g.ndata['label'].unique().shape[0],log_training=False)
    acc_list.append(acc)
  return acc_list


def run_full_simulation(g,mode='rewiring',k=50,extreme_scale = 10,num_saves=20,N_max=10000):

  """
  Run the full simulation, generating a list of graphs, and testing the performance of a GNN on a graph classification task, 
  and calculating the geodesics of the graphs produced.

  Args:
      g: dgl graph
      k: number of graphs to try to generate
      extreme_scale: factor to change the eigenvalues by to achieve extreme (low/high) homophily

  Returns:
      g_list: list of dgl graphs
      acc_list: list of GNN accuracy values
      mean_spl_list: mean shortest path length between all nodes in the graph
      mean_intra_spl_list: mean shortest path length between nodes of the same class

  """
  print('Generating synthetic graphs...')
  if mode == 'SBM':
    g_list = gen_synthetic_SBM_graph_list(g,k=k,extreme_scale = extreme_scale)
  
  elif mode == 'rewiring':
    g_list_homo = gen_synthetic_rewired_graph(g, h=1,num_saves=num_saves,N_max=N_max)
    g_list_hetero = gen_synthetic_rewired_graph(g, h=0,num_saves=num_saves,N_max=N_max)
    g_list = g_list_homo+g_list_hetero
  
  print('Testing GNN performance...')
  acc_list = test_gnn_vs_homophily(g_list)
  print('Calculating geodesics...')
  mean_spl_list,mean_intra_spl_list = calculate_geodesics_homophily(g_list)

  return g_list,acc_list,mean_spl_list,mean_intra_spl_list

def calculate_homophily(g_list):
  """
  Calculate the homophily of a list of graphs.

  Args:
      g_list: list of dgl graphs

  Returns:
      h_list: list of homophily values

  """
  h_list = []
  for g in tqdm(g_list):
    h_list.append(class_homophily(g))
  return h_list


def spl_rewiring(g0,model):
    """
    Rewire a graph based on the shortest path length between nodes of the same class to increase 
    homophily. Nodes of the same class that have an spl closer to the mean intra-class spl of the 
    graph are connected, and nodes of different classes that are connected are disconnected.

    Args:
        g0: dgl graph
        model: model to use to calculate label predictions

    Returns:
        g: dgl graph rewired

    """
    g = g0.clone()
    labels = g.ndata['label']
    pred_labels = model(g,g.ndata['feat']).argmax(1)
    print('Calculating shortest path lengths distribution...')
    
    # calculate the shortest path length between nodes of the same class
    spl_list = []
    for c in tqdm(g.ndata['label'].unique()):
        nodes = g.nodes()[(pred_labels==c).numpy()]
        spl_list.append(spl_between_nodes(g,nodes,n_max=500))
    
    spl_arr = np.concatenate(spl_list)
    lengths = spl_arr[:,2]

    std_spl = np.std(lengths)
    mean_intra_spl = np.mean(lengths)
    # calculate mask of node pairs that have a spl close to the mean intra-class spl
    connect_mask = np.abs(spl_arr[:,2]-mean_intra_spl)<std_spl

    pairs_to_connect = spl_arr[connect_mask][:,0:2]

    u_connect, v_connect = np.array(pairs_to_connect).T
    g.add_edges(u_connect, v_connect)
    
    print(f'added {len(pairs_to_connect)} edges')

    remove_eids = []
    print('rewiring edges...')
    for i in tqdm(range(len(u_connect))):
        u,v = u_connect[i], v_connect[i]
        neighbour_pairs = np.concatenate((torch.stack(g.in_edges(u)).numpy().T,
                        torch.stack(g.in_edges(v)).numpy().T))

        for neighbour_pair in neighbour_pairs:
            if labels[neighbour_pair[0]] != labels[neighbour_pair[1]]:
                remove_eids.append(g.edge_id(*neighbour_pair))
                
                break
    print(f'Tried removing {len(remove_eids)} edges')
    print(f'final number of edges before: {g.number_of_edges()}')
    g = dgl.remove_edges(g,remove_eids) #######this doesnt work?
    print(f'final number of edges after: {g.number_of_edges()}')
    print(f'Number entries of A.T != A: {g.adj().to_dense().T != g.adj().to_dense().sum()}')
    
    return g