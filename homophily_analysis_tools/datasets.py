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


def relable(labels):
  """
  Relable graph labels uniquely as increasing integers
  Args:
    labels: torch tensor of labels
  Returns:
    relabled labels
  """
  new_l = 0
  new_labels = -1*torch.ones(labels.shape[0])
  for i,l in enumerate(labels):
    if new_labels[i]==-1:
      new_labels[labels==l]=new_l
      new_l +=1
  return new_labels.long()


def pyg_to_dgl_graph_converter(pyg_g):
  """
  Convert a pytorch geometric graph to a dgl graph
  Args:
    pyg_g: pytorch geometric graph
  Returns:
    dgl graph
  """
  u,v = pyg_g.edge_index
  f = pyg_g.x
  labels = pyg_g.y
  g = dgl.graph((u,v))
  g.ndata['feat'] = f
  g.ndata['label'] = relable(labels)
  return g


def add_train_val_test_masks(g):
  """ 
  Add train, val, test masks to graph for the purpose of training a classifier
  Args:
    g: dgl graph
  Returns:
    g: dgl graph with train, val, test masks

  """
  data_len = g.ndata['label'].shape[0]
  ind = torch.tensor(range(data_len))
  perm = torch.randperm(data_len)
  split = (int(0.8*data_len), int(0.9*data_len),int(data_len))

  train_mask = torch.tensor([i in ind[:split[0]] for i in ind])[perm]
  val_mask = torch.tensor([i in ind[split[0]:split[1]] for i in ind])[perm]
  test_mask = torch.tensor([i in ind[split[1]:] for i in ind])[perm]

  g.ndata.update({'test_mask':test_mask})
  g.ndata.update({'train_mask':train_mask})
  g.ndata.update({'val_mask':val_mask})
  return g


class NewCoraGraphDataset(dgl.data.CoraGraphDataset):
    """
    Dataset class wrapper of CoraGraphDataset to add train, val, test masks to graph
    Args:
      None
    Returns:
      None
    
    """
    def __init__(self):
      super().__init__()

      add_train_val_test_masks(self[0])


class NewCiteseerGraphDataset(dgl.data.CiteseerGraphDataset):
    """
    Dataset class wrapper of CiteseerGraphDataset to add train, val, test masks to graph
    Args:
      None
    Returns:
      None
    
    """
    def __init__(self):
      super().__init__()

      add_train_val_test_masks(self[0])


class NewPubmedGraphDataset(dgl.data.PubmedGraphDataset):
    """
    Dataset class wrapper of PubmedGraphDataset to add train, val, test masks to graph
    Args:
      None
    Returns:
      None
    
    """

    def __init__(self):
      super().__init__()

      add_train_val_test_masks(self[0])


class NewPPI0GraphDataset(dgl.data.PPIDataset):
    """
    Dataset class wrapper of PPI0GraphDataset to add train, val, test masks to graph
    Args:
      None
    Returns:
      None
    
    """

    def __init__(self):
      super().__init__()
      for graph in tqdm(self):
        graph = add_train_val_test_masks(graph)


class ActorGraphDataset(dgl.data.DGLDataset):
  """
  Dataset class wrapper of ActorGraphDataset to add train, val, test masks to graph 
  and to convert to dgl graph
  Args:
    None
  Returns:
    None

  """


  def __init__(self):
      super().__init__(name='ActorGraphDataset')
      ActorDataset = torch_geometric.datasets.Actor(os.getcwd())
      self.dg_dataset = []
      for i,pyg_graph in enumerate(tqdm(ActorDataset)):
        graph = pyg_to_dgl_graph_converter(pyg_graph)
        graph = add_train_val_test_masks(graph)
        self.dg_dataset.append(graph)
    
  def __getitem__(self,i):
    return self.dg_dataset[i]

class LINKXDatasetGraphDataset():
  """
  Dataset class wrapper of LINKXDatasetGraphDataset to add train, val, test masks to graph
  and to convert to dgl graph
  Args:
    None
  Returns:
    None


  """
  
  def __init__(self):
      names = ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55"] #, "genius"
      #if name not in names:
      #  raise NotImplementedError('Please select one of "penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"')
      pyg_dataset = [torch_geometric.datasets.LINKXDataset(os.getcwd(),name)[0] for name in names]
      self.dgl_dataset = []
      for i,pyg_graph in enumerate(tqdm(pyg_dataset)):
        graph = pyg_to_dgl_graph_converter(pyg_graph)
        graph = add_train_val_test_masks(graph)
        self.dgl_dataset.append(graph)
    
  def __getitem__(self,i):
    return self.dgl_dataset[i]


class AmazonDatasetGraphDataset():
  """
  Dataset class wrapper of AmazonDatasetGraphDataset to add train, val, test masks to graph
  and to convert to dgl graph
  Args:
    None
  Returns:
    None

  """

  
  def __init__(self):
      names = ["Computers", "Photo"]
      pyg_dataset = [torch_geometric.datasets.Amazon(os.getcwd(),name)[0] for name in names]
      self.dgl_dataset = []
      for i,pyg_graph in enumerate(tqdm(pyg_dataset)):
        graph = pyg_to_dgl_graph_converter(pyg_graph)
        graph = add_train_val_test_masks(graph)
        self.dgl_dataset.append(graph)
    
  def __getitem__(self,i):
    return self.dgl_dataset[i]


def gen_complete_homophilic_graph(n,k):
    """
    Generates a complete homophilic graph with n nodes and k classes of labels removing all 
    edges between nodes of different classes
    Args:
      n: number of nodes
      k: number of classes
    Returns:
      dgl graph
    """

    # create a complete graph with n nodes, with k labels and homophily 1
    # a complte graph is where every node connects to every other node through an edge
    # remove all edges between nodes of a different label

    g = dgl.DGLGraph()
    g.add_nodes(n)
    
    # add edges between all nodes
    # get all permutations of all nodes 

    p = list(itertools.permutations(g.nodes(),2))
    g.add_edges([x[0] for x in p],[x[1] for x in p])

    g.ndata['label'] = torch.randint(k,(n,))
    
    eids = []
    for i in range(n):
        for j in range(n):
            if i != j:
                if g.ndata['label'][i] != g.ndata['label'][j]:
                    eids.append(g.edge_id(i,j))
    g.remove_edges(eids)

    return g


def gen_complete_heterophilic_graph(n,k):
    """
    Generates a complete heterophilic graph with n nodes and k classes of labels, removing 
    all edges between nodes of the same label

    Args:
      n: number of nodes
      k: number of classes
    Returns:
      dgl graph
    """
    #create a complete graph with n nodes, with k labels and homophily 0
    # a complte graph is where every node connects to every other node through an edge
    # rempove all edges between nodes of the same label

    g = dgl.DGLGraph()
    g.add_nodes(n)

    # add edges between all nodes
    # get all permutations of all nodes 

    p = list(itertools.permutations(g.nodes(),2))
    g.add_edges([x[0] for x in p],[x[1] for x in p])

    g.ndata['label'] = torch.randint(k,(n,))
    
    eids = []
    for i in range(n):
        for j in range(n):
            if i != j:
                if g.ndata['label'][i] == g.ndata['label'][j]:
                    eids.append(g.edge_id(i,j))
    g.remove_edges(eids)

    return g