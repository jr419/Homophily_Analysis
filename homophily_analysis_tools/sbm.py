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


def infer_Z(y):
  """
  Infer the categorization matrix Z from the labels y
  Args:
    y: labels
  Returns:
    Z: categorization matrix
  """
  n = y.shape[0]
  v = np.arange(n)
  Z = torch.sparse_coo_tensor((v,y),torch.ones(n)).to_dense()
  return Z


def sbm(g,p=1):
  """
  MLE implementation of SBM coarsening of a graph
  Args:
    g: dgl graph
    p: p-hop neighbourhoods to consider
  Returns:
    S: Class Balanced Affinity matrix
    B: Affinity matrix of classes
    Pi: Class Distribution diagonal matrix
  """
  A = g.adj().to_dense()
  y = g.ndata['label']
  n = y.shape[0]
  k = y.unique().shape[0]
  Z = infer_Z(y)

  u = torch.ones(n)
  U = torch.outer(u,u)

  Pi = (Z.T@u).diag()/n
  A_p = torch.matrix_power(A,p)
  B = Z.T@A_p@Z*n/(Z.T@U@Z)
  return B, Pi


def sbm_dc(g,mode='sym',p=1):
  """
  Implementation of the degree correlated SBM coarsening of a graph.

  Args:
    g: dgl graph
    mode: 'rw' for random walk, 'sym' for symmetric
    p: p-hop neighbourhoods to consider
  Returns:
    M: Affinity matrix of classes, equals the number of edges between nodes of any two classes
    Pi: Class Distribution diagonal matrix
    theta: blockwise degree distribution
  """

  A = g.adj().to_dense()
  y = g.ndata['label']
  n = y.shape[0]
  k = y.unique().shape[0]
  d = torch.sum(A,dim=1)
  u = torch.ones(n)
  U = torch.outer(u,u)

  Z = infer_Z(y)
  d_inv = 1/d.float()
  d_inv[d==0] = 1
  d_inv_sqrt = d_inv.sqrt()
  d_mean = ((d**p).mean())**(1/p)
  Pi = (Z.T@u).diag()/n

  if mode == 'sym':
    A_hat = d_inv_sqrt[:,None]*A*d_inv_sqrt[None,:]
  elif mode == 'rw':
    A_hat = d_inv[:,None]*A
  elif mode == 'edge':
    A_hat = A/d_mean
  else:
    raise ValueError('mode must be either "edge","rw", "sym"')

  A_hat_p = torch.matrix_power(A_hat,p)
  M = Z.T@A_hat_p@Z*n/(Z.T@U@Z)

  kappa = torch.stack([torch.sum(d[y == c]**p) for c in g.ndata['label'].unique()])/(Z.T@u)
  return M, Pi, kappa
