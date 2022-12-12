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

from homophily_analysis_tools.sbm import sbm_dc, sbm


def edge_homophily(g,p=1,class_normalised=True):
  """
  Calculate the edge homophily of a graph, defined as the average of the number of edges between nodes of the same class. 
  Optionally take in a power p to calculate the homophilly over a p-hop neighbourhood.

  Args:
    g: dgl graph
    p: size of neighbourhood to calculate homophily over

  Returns:
    h: edge homophily of graph
  """
  B, Pi = sbm(g,p)
  pi = Pi.diag()

  if class_normalised:
    h = B.trace()/B.sum()
  else:
    h = (Pi@B@Pi).trace()/(pi.T@B@pi)
  return h


def homophily(g,p=1,type='rw',class_normalised=False):
  """
  Calculate the edge homophily of a graph, defined as the average of the number of edges between nodes of the same class. 
  Optionally take in a power p to calculate the homophilly over a p-hop neighbourhood.

  Args:
    g: dgl graph
    p: size of neighbourhood to calculate homophily over
    adj_type: adjacency matrix type to use, either 'sym', 'rw' or 'edge'

  Returns:
    h: edge homophily of graph
  """
  B, Pi,_ = sbm_dc(g,mode=type,p=p)
  pi = Pi.diag()
  n = g.number_of_nodes()

  if class_normalised:
    h = B.trace()
  else:
    h = (Pi@B@Pi).trace()
  return h


def class_homophily(g,p=1):
  """
  Calculate the full class homophily of a graph, defined as the average of the number of edges between nodes of the same class, 
  normalised by the *square* of the class size, so as to make it independent of the number of class distribution entirely.

  Optionally take in a power p to calculate the homophilly over a p-hop neighbourhood.

  Args:
    g: dgl graph
    p: size of neighbourhood to calculate homophily over

  Returns:
    h: class homophily of graph
  """
  B, Pi = sbm(g)
  k = B.shape[0]
  pi = Pi.diag()
  B_p = torch.matrix_power(B,p)
  h = ((B_p).trace())/(B_p.sum())
  return h


def get_eigs(g,p=1):
  """
  Calculate the eigenvalues of the SBM class normalised affinity matrix (Pi B) of a graph, optionally taking in a power p to calculate the
  eigenvalues of the p-hop affinity matrix.

  Args:
    g: dgl graph
    p: size of neighbourhood to calculate eigenvalues of affinity matrix over

  Returns:
    eigs: eigenvalues of affinity matrix
  """

  B,Pi = sbm(g)
  S = (Pi**(1/2))@B@(Pi**(1/2))
  S_p = torch.matrix_power(S,p)
  l,_ = torch.linalg.eigh(S_p)
  return l


def graph_summary(g_list):
  """
  Calculate a summary of the informatuion about the graph, including the homophily, 
  mean degree, class distribution, and eigenvalues of a list of graphs.

  Args:
    g_list: list of dgl graphs

  Returns:
    None
  """
  for i,g in enumerate(g_list):
    d = g.in_degrees().float().mean()
    l = get_eigs(g)
    B,Pi = sbm(g)
    class_dist = Pi.diag().numpy()
    print('='*50+f"(GRAPH {i+1})"+'='*50)
    print('Eigenvalues: ', list(l.numpy()))
    print('Leading Eigenvalue: ', l[-1].item())
    print('Mean Degree: ', d.item())
    print('Class Dist: ',class_dist)


    print('Edge Homophily (p=1): ', homophily(g,type='edge').item())
    print('Node Homophily (p=1): ', homophily(g,type='sym').item())
    print('Edge Homophily (p=2): ', homophily(g,type='edge',p=2).item())
    print('Node Homophily (p=2): ', homophily(g,type='sym',p=2).item())

