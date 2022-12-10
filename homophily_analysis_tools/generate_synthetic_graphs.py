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

from homophily_analysis_tools.datasets import add_train_val_test_masks
from homophily_analysis_tools.homophily_metrics import class_homophily, edge_homophily, node_homophily
from homophily_analysis_tools.sbm import infer_Z, sbm_dc, sbm

def generate_Z(Pi,n):
  """
  Generate a categorization matrix Z radomly from a class distribution matrix Pi
  
  Args:
    Pi: Class Distribution diagonal matrix
    n: number of nodes
  Returns:
    Z: categorization matrix

  """
  k = Pi.shape[0]
  u = np.ones(n)
  Pi_repeated = np.outer(u,Pi.diag())
  Z = stats.bernoulli.rvs(Pi_repeated, size=(n,k))

  return Z


def coarsen(A,y,d):
  """
  Coarsen a graph using the SBM model

  Args:
    A: Adjacency matrix
    y: labels
    d: average degree of graph
  Returns:
    B: Affinity matrix of classes
    Pi: Class Distribution diagonal matrix
  """

  n = y.shape[0]
  Z = infer_Z(y)
  u = torch.ones(n)
  U = torch.outer(u,u)

  B = Z.T@A@Z/(Z.T@U@Z)
  Pi = (Z.T@u).diag()/n

  kappa = Z.T@d/(Z.T@u)
  theta = d/(Z@kappa)
  
  return B, Pi, theta


def decoarsen(B, Pi, y, theta,d ):
  """
  Decoarsen a graph from a degree correlated SBM model. Due to the change in homophily, the degree 
  distributuion needs to be readjusted, after the expected adjacency matrix is calculated.

  Args:
    B: Affinity matrix of classes
    Pi: Class Distribution diagonal matrix
    y: labels
    theta: blockwise normalised degree distribution
    d: average degree of graph
  
  Returns:
    A: Adjacency matrix
  
  """
  n = y.shape[0]
  Z = infer_Z(y)

  # calculate the expected adjacency matrix
  
  A_p = theta[:,None] *(Z@B@Z.T) * theta[None,:]

  # rescale the degree distribution
  d_pred = A_p.sum(1)
  d_sf = (d/d_pred).sqrt()
  d_sf[d_pred==0] = 1
  A_p = d_sf[:,None] * A_p * d_sf[None,:]
  A = stats.poisson.rvs(mu=A_p, size=(n,n))

  
  return A


def make_graph(A,y):
  """ 
  Make a dgl graph from an adjacency matrix and labels

  Args:
    A: Adjacency matrix
    y: labels
  Returns:
    g: dgl graph

  """
  n = y.shape[0]
  g = dgl.graph(torch.where(torch.tensor(A)>0),num_nodes=n)
  g.ndata['label'] = torch.tensor(y)
  return g


def change_homophily_spectral(B,Pi,d,h):
  """
  Change the homophily of a graph by tuning the eigenvalues of the SBM model, by
  scaling the non-leading eigenvalues by a constant factor h.

  Args:
    B: Affinity matrix of classes
    Pi: Class Distribution diagonal matrix
    d: average degree of graph
    h: hyperparameter
  Returns:
    B_new: Affinity matrix of classes
  
  """
  b_sum = B.sum()
  l,Q = torch.linalg.eigh(B)
  l[:-1] = h*l[:-1]
  B_new = Q@l.diag()@Q.T
  sf = b_sum/B_new.sum()
  B_new = sf*B_new
  return B_new


def change_homophily(B,Pi,d,h):
  """
  Change the homophily of a graph by tuning the eigenvalues of the SBM model, by
  simply scaling the diagonal of the affinity matrix by a constant factor h.

  Args:
    B: Affinity matrix of classes
    Pi: Class Distribution diagonal matrix
    d: average degree of graph
    h: hyperparameter
  Returns:
    B_new: Affinity matrix of classes
  
  """
  b_sum = B.sum()
  pi = Pi.diag()
  # scale the diagonal elements of B by h
  B_new = B.clone()
  diag_mask = torch.eye(B.shape[0],dtype=torch.bool)
  B_new[diag_mask] = h*B_new[diag_mask]
  sf = b_sum/B_new.sum()
  B_new = B_new*sf

  return B_new


def gen_synthetic_graph_spectral(g0,d,h):
  """ 
  Generate a synthetic graph from a given graph using the SBM model, and change the homophily of the graph
  by scaling the non-leading eigenvalues by a constant factor h. 
  
  Optionally take in a desired average degree or full degree sequence to normalise the coarsening of the 
  graph and or the decoarsening of the graph.

  Additonally if the resultant adjacency matrix has entries greater than 1 or less than 0, None is returned.

  Args:
    g0: dgl graph
    h: hyperparameter to tune homophily
    d_in: average degree of graph or full degree sequence for normalising coarsening
    d_out: average degree of graph or full degree sequence for normalising decoarsening
  Returns:
    g_synthetic: dgl graph
  
  """
  
  g = g0.clone()
  n = g.number_of_nodes()
  A = g.adj().to_dense()
  
  y = g.ndata['label']
  k = y.unique().shape[0]
  feat = g.ndata['feat']
  B, Pi,theta = coarsen(A,y,d)

  B = change_homophily(B,Pi,d,h)

  A = decoarsen(B, Pi, y,theta,d)


  if A.any():
    g_synthetic = make_graph(A,y)
    g_synthetic.ndata['feat'] = feat
    add_train_val_test_masks(g_synthetic)
    return g_synthetic
  else:
    return None


def gen_synthetic_SBM_graph_list(g,k=50,extreme_scale = 10):
  """
  Generate a list of graphs by randomly by coarsening to an SBM distribution of a given seed graph, and changing it's homophily
  by tuning the eigenvalues of the SBM. 

  Small homophily eigenvalues are associated with low homophily, and large eigenvalues are associated with high homophily.

  To achieve extreme low homophily, we change the eigenvalues by a smaller factor to avoid getting adjecency probability matricies with
  entries outside of the range [0,1].

  Args:
      g: dgl graph
      do_full_spl: whether to calculate the mean shortest path length between all nodes in the graph
      k: number of graphs to try to generate
      extreme_scale: factor to change the eigenvalues by to achieve extreme (low/high) homophily

  Returns:
      mean_spl_list: mean shortest path length between all nodes in the graph
      mean_intra_spl_list: mean shortest path length between nodes of the same class
      mean_eigenvalue_list: mean eigenvalue of the graphs
      g0_list: list of dgl graphs

  
  """
  g0_list = []
  
  for i in tqdm(range(4*k)):
    if i <k:
        g0 = gen_synthetic_graph_spectral(g,h=i/(extreme_scale*k),d=g.in_degrees().float())
    elif i >= k and i <= 3*k:
        g0 = gen_synthetic_graph_spectral(g,h=(i-k)/(2*k),d=g.in_degrees().float())
    else:
        g0 = gen_synthetic_graph_spectral(g,h=1+(extreme_scale-1)*(i-3*k)/k,d=g.in_degrees().float())


    g0_list.append(g0)

  return g0_list


def gen_synthetic_rewired_graph(g0,h=1,N_max=10000,num_saves=20):
  """
  Generate a synthetic graph from a given graph by adding and removing edges, in a degree preserving manner,
  to change the homophily of the graph. To increase the homophily of the graph, edges are added between nodes
  of the same class, and edges are removed between nodes of different classes. To decrease the homophily of the
  graph, edges are added between nodes of different classes, and edges are removed between nodes of the same class.

  Args:
    g0: dgl graph
    h: desired homophily
    N_max: maximum number of iterations to try to generate a graph
    interval: interval to calculate the current homophily of the graph
  
  Returns:
    g_list: list of dgl graphs generated over the iterations
    h_list: list of homophilies of the graphs generated over the iterations

  """
  interval = N_max//num_saves

  g = g0.clone()
  h0 = class_homophily(g)
  n = g.number_of_nodes()
  classes = g.ndata['label'].unique()

  cnt = 0
  h_list = []
  g_list = []

  if h<h0:
    mode = -1
  else:
    mode = 1
  
  while mode * (h0-h)<0 and cnt<N_max:
    edges = torch.stack(g.edges('all'))
    pi = np.array([(g.ndata['label']==c).float().mean() for c in classes])

    c = np.random.choice(classes,p=pi) 
    k_c = (g.ndata['label']==c).sum()
    k_n = 0
    n_v = 0
    cnt2 = 0
    while (k_n==0 or n_v-k_n==0) and cnt2<n:
      u = g.nodes()[g.ndata['label']==c][np.random.randint(0,k_c)]
      
      neighbour_v,_,neighbour_eid = g.in_edges(u,'all')
      k_n = (g.ndata['label'][neighbour_v]==c).sum()
      n_v = neighbour_v.shape[0]
      cnt2 +=1
    
    if k_n==0 or n_v-k_n==0:
      return g_list

    if mode<0:
      eid = neighbour_eid[g.ndata['label'][neighbour_v]==c][np.random.randint(0,k_n)]
      v = g.nodes()[g.ndata['label']!=c][np.random.randint(0,n-k_c)]
    
    if mode>0:
      eid = neighbour_eid[g.ndata['label'][neighbour_v]!=c][np.random.randint(0,n_v-k_n)]
      v = g.nodes()[g.ndata['label']==c][np.random.randint(0,k_c)]
    
    g = dgl.remove_edges(g, eid)
    g.add_edges(u,v)
    
    if cnt%interval==0:

      h0 = class_homophily(g)
      h_list.append(h0)
      g_list.append(g)
      print(f"{cnt//interval+1}/{N_max//interval}) {h0}")
    cnt +=1
  return g_list