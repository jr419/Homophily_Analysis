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


import itertools

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


def sbm(g):
  """
  MLE implementation of SBM coarsening of a graph
  Args:
    g: dgl graph
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
  
  B = Z.T@A@Z*n/(Z.T@U@Z)
  return B, Pi


def sbm_dc(g):
  """
  Implementation of the degree correlated SBM coarsening of a graph.

  Args:
    g: dgl graph
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
  d_inv_sqrt = (1/d.float().sqrt())
  d_inv_sqrt[d==0] = 1
  Pi = (Z.T@u).diag()/n
  A_hat = d_inv_sqrt[:,None]*A*d_inv_sqrt[None,:]
  M = Z.T@A_hat@Z*n/(Z.T@U@Z)
  # theta is given by the degree of the node i divided by the sum of all degrees of the smae label as node i
  kappa = torch.stack([torch.sum(d[y == c]) for c in g.ndata['label'].unique()])/(Z.T@u)
  return M, Pi, kappa


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
  B, Pi = sbm(g)
  pi = Pi.diag()
  B_p = torch.matrix_power(B,p)
  if class_normalised:
    h = B_p.trace()/B_p.sum()
  else:
    h = (Pi@B_p@Pi).trace()/(pi.T@B_p@pi)
  return h


def node_homophily(g,p=1,class_normalised=False):
  """
  Calculate the edge homophily of a graph, defined as the average of the number of edges between nodes of the same class. 
  Optionally take in a power p to calculate the homophilly over a p-hop neighbourhood.

  Args:
    g: dgl graph
    p: size of neighbourhood to calculate homophily over

  Returns:
    h: edge homophily of graph
  """
  B, Pi,kappa = sbm_dc(g)
  pi = Pi.diag()
  B_p = torch.matrix_power(B,p)
  if class_normalised:
    h = B_p.trace()/B_p.sum()
  else:
    h = (Pi@B_p@Pi).trace()/(pi.T@B_p@pi)
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


    print('Edge Homophily: ', edge_homophily(g).item())
    print('Class Normalised Homophily: ', class_homophily(g).item())

    
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


def train(g, model, log_training=True,n_epochs=200):
    """
    Train a GNN model on a graph. Optionally log the training loss.

    Args:
        g: dgl graph
        model: GNN model
        log_training: whether to log the training loss
        n_epochs: number of epochs to train for

    Returns:
        accuracy of model on graph

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_acc = 0
    best_test_acc = 0
    g = g.to(device)
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].to(device)

    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(n_epochs):
        # Forward
        logits = model(g, features)
        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = f1_score(labels[train_mask].cpu(), pred[train_mask].cpu(), average='macro') #(pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = f1_score(labels[val_mask].cpu(), pred[val_mask].cpu(), average='macro') #(pred[val_mask] == labels[val_mask]).float().mean()
        test_acc_per_node = f1_score(labels[test_mask].cpu(), pred[test_mask].cpu(), average='macro') #(pred[test_mask] == labels[test_mask]).float()
        test_acc = test_acc_per_node.mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if (best_val_acc+best_test_acc)/2 < (val_acc+test_acc)/2:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0 and log_training:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc,best_test_acc))
    return (best_val_acc+best_test_acc)/2


class GCN(nn.Module):
    """ 
    Graph Convolutional Network model

    Args:
        in_feats: number of input features
        h_feats: number of hidden features
        num_classes: number of classes
    

    """
    def __init__(self, in_feats, h_feats, num_classes,num_layers=2):
        super(GCN, self).__init__()
        self.conv_in = GraphConv(in_feats, h_feats,allow_zero_in_degree=True)
        self.conv_out = GraphConv(h_feats, num_classes,allow_zero_in_degree=True)
        self.inter_layers = nn.ModuleList([GraphConv(h_feats, h_feats,allow_zero_in_degree=True) for _ in range(num_layers-2)])

    def forward(self, g, in_feat):
        """
        Forward pass of the model

        Args:
            g: dgl graph
            in_feat: input features

        Returns:
            h: output features
        """
        in_feat = torch.tensor(in_feat, dtype=torch.float)
        h = self.conv_in(g, in_feat)
        h = F.relu(h)
        for layer in self.inter_layers:
            h = layer(g, h)
            h = F.relu(h)
        h = self.conv_out(g, h)
        #h = F.softmax(h)
        return h
            

def train_GCN_g(g,num_classes, log_training=True,return_model=False,num_layers=2):
    """
    Train a GCN model on a graph. Optionally log the training loss. Optionally return the model.

    Args:
        g: dgl graph
        num_classes: number of classes
        log_training: whether to log the training loss
        return_model: whether to return the model

    Returns:
        accuracy of model on graph

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn_model = GCN(g.ndata['feat'].shape[1], 256, num_classes,num_layers=num_layers).to(device)
    if return_model:
        return train(g, gcn_model, log_training),gcn_model
    else:
        return train(g, gcn_model, log_training)


class MLP(nn.Module):
    """
    Multi-layer perceptron model

    Args:
        input_dim: number of input features
        output_dim: number of hidden features

    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 150)
        self.hidden_fc = nn.Linear(150, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, g, in_feat):
        in_feat = torch.tensor(in_feat, dtype=torch.float)
        h_1 = F.relu(self.input_fc(in_feat))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = F.softmax(self.output_fc(h_2))


        return y_pred


def train_MLP_g(g,num_classes, log_training=True):
    """
    Train a MLP model on a graph. Optionally log the training loss.

    Args:
        g: dgl graph
        num_classes: number of classes
        log_training: whether to log the training loss

    Returns:
        accuracy of model on graph

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp_model = MLP(g.ndata['feat'].shape[1], num_classes).to(device)
    return train(g, mlp_model, log_training=log_training)


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



def plot_mean_spl_comparison(g0_list,mean_spl_list=None,mean_intra_spl_list=None,p=1):
  """
  Plot the mean shortest path length between all nodes in the graph, and the mean shortest path length between nodes of the same class
  as a function of the homohiply of the graphs produced.

  Args:
      mean_eigenvalue_list: mean eigenvalue of the graphs
      g0_list: list of dgl graphs
      mean_spl_list: mean shortest path length between all nodes in the graph
      mean_intra_spl_list: mean shortest path length between nodes of the same class
      p: p-th hop neighborhood homohilly to plot

  Returns:
      None

  """
  h_list = []
  for g in g0_list:
    h_list.append(class_homophily(g,p=p))
  
  d_list = []
  n = g0_list[0].number_of_nodes()
  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111)
  ax.grid(color='gray', linestyle='-', linewidth=0.1)
  ax.set_xlabel(f'Homophily Level (p={p})')
  ax.set_ylabel('Mean shortest path length')
  ax.set_title(f'Mean shortest path length vs Homophily Level (p={p})')

  if mean_spl_list is not None:
    ax.plot(h_list,mean_spl_list,'o',label='Mean shortest path length')

  if mean_intra_spl_list is not None:
    ax.plot(h_list,mean_intra_spl_list,'o',label='Mean intra class shortest path length')

  ax.legend()
  plt.show()

  return


def plot_test_gnn_vs_homophily(g_list,acc_list,p = 1):
  """
  Plot the precomputed performance of a GNN on a graph classification task, as a function of precomputed homophily of the graphs produced.

  Args:
      g_list: list of dgl graphs
      acc_list: list of accuracies
      p: p-th hop neighborhood homohilly to plot

  Returns:
      None

  """

  h_list = []
  for g in g_list:
    h_list.append(class_homophily(g, p=p))
  
  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111)
  ax.grid(color='gray', linestyle='-', linewidth=0.1)
  ax.set_xlabel(f'Homophily Level (p={p})')
  ax.set_ylabel('GNN Accuracy')
  ax.set_title(f'GNN Accuracy as a function of Homophily Level (p={p})')

  ax.plot(h_list,acc_list,'o',label='Accuracy')

  ax.legend()
  plt.show()

  return

def plot_test_gnn_vs_spl(spl_list,acc_list):
  """
  Plot the performance of a GNN on a graph classification task, as a function of precomputed mean shortest path lengths 
  of nodes considered in the graph.

  Args:
      spl_list: list of mean shortest path lengths
      acc_list: list of accuracies  

  Returns:
      None

  """

  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111)
  ax.grid(color='gray', linestyle='-', linewidth=0.1)
  ax.set_xlabel('Mean shortest path length')
  ax.set_ylabel('GNN Accuracy')
  ax.set_title('GNN Accuracy as a function of mean shortest path length')

  ax.plot(spl_list,acc_list,'o',label='Accuracy')

  ax.legend()
  plt.show()
  
  return

def plot_homophily_degree(g0_list,p=1):
  """
  Plot the mean and max degree of the nodes in the graph, as a function of the homophily of the graphs produced.

  Args:
      g0_list: list of dgl graphs

  Returns:
      None

  """
  h_list = []
  d_mean_list = []
  d_max_list = []
  for g in g0_list:
    h_list.append(class_homophily(g,p=p))
    d_mean_list.append(g.in_degrees().float().mean())
    d_max_list.append(g.in_degrees().float().max())
  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111)
  ax.grid(color='gray', linestyle='-', linewidth=0.1)
  ax.set_xlabel(f'Homophily Level (p={p})')
  ax.set_ylabel('Mean degree')
  ax.set_title(f'Mean degree as a function of homophily (p={p})')

  ax.plot(h_list,d_mean_list,'o',label='Mean degree')

  ax.legend()
  plt.show()

  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111)
  ax.grid(color='gray', linestyle='-', linewidth=0.1)
  ax.set_xlabel('Homophily')
  ax.set_ylabel('Max degree')
  ax.set_title('Max degree as a function of homophily')

  ax.plot(h_list,d_max_list,'o',label='Max degree')

  ax.legend()
  plt.show()

  return


def plot_degree_distribution_comparison(g_list):
  """ 
  Plot the degree distribution of the graphs in g_list

  Args:
      g_list: list of dgl graphs

  Returns:
      None

  """
  
  # plot a bar chart with the y axis representing the frequency of each degree of a graph g
  # plot charts for all the graphs in the list g_list
  d0 = max([g.in_degrees().float().max() for g in g_list])
  fig = plt.figure(figsize=(10,6))
  for i,g in enumerate(g_list):
    d_list = g.in_degrees().numpy()
    d_list = d_list[d_list>0]
    d_list = d_list[d_list<100]
    
    #dynamically change the intensity of the color of the bar chart based on the average degree of the graph
    # change intensity by changing alpha 
    alpha = 0.5+0.25*(1-g.in_degrees().float().max()/d0)
    plt.hist(d_list,bins=100,alpha=alpha.item(),label='h = '+str(round(class_homophily(g).item(),2)),density=True)

    #indicate the average degree of each graph on the x-axis with a dashed red line
    plt.axvline(x=g.in_degrees().float().mean(),color='C'+str(i),linestyle='--')
  
  plt.legend()
    

  plt.xlabel('Degree')
  plt.ylabel('Frequency')
  
  return


#plot the difference of the mean shotest path lengths between the same classes and the mean 
# shortest path lengths between all nodes as a function of homophily

def plot_mean_spl_difference(g_list,mean_spl_list,mean_intra_spl_list):
    """ 
    Plot the difference of the mean shotest path lengths between the same classes and the mean 
    shortest path lengths between all nodes as a function of homophily
    
    Args:
        mean_eigenvalue_list: list of mean eigenvalues of the adjacency matrices of the graphs in g_list
        g_list: list of dgl graphs
        mean_spl_list: list of mean shortest path lengths of the graphs in g_list
        mean_intra_spl_list: list of mean shortest path lengths between nodes of the same class of the graphs in g_list
    
    Returns:
        None
    
    """
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.grid(color='gray', linestyle='-', linewidth=0.1)
    ax.set_xlabel('Homophily')
    ax.set_ylabel('Mean shortest path - Mean intra shortest path')
    h_list = [class_homophily(g).item() for g in g_list]
    mean_spl_arr = np.array(mean_spl_list)
    mean_intra_spl_arr = np.array(mean_intra_spl_list)
    ax.plot(h_list,mean_spl_arr-mean_intra_spl_arr,'o',label='Mean shortest path length difference')
    ax.legend()
    plt.show()
    
    return


def do_plots_compact(g_list,g_seed,mean_spl_list,mean_intra_spl_list,acc_list,p=1):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].set_title('Mean shortest path length')
    ax[0].set_xlabel(f'Homophily Level (p={p})')
    ax[0].set_ylabel('Mean shortest path length')
    h_list = [class_homophily(g,p=p) for g in g_list]

    h_seed = class_homophily(g_seed,p=p)
    mean_spl_seed, mean_intra_spl_seed = calculate_geodesics_homophily([g_seed])
    acc_seed = test_gnn_vs_homophily([g_seed])
    
    
    ax[0].plot(h_list,mean_spl_list,'.',label='Mean shortest path length')
    # add the seed graph to the plot, with a different marker in red
    ax[0].plot(h_seed,mean_spl_seed,'*',color='red',label='Seed graph')

    ax[0].plot(h_list,mean_intra_spl_list,'.',label='Mean intra-class shortest path length')
    # add the seed graph to the plot, with a different marker in red
    ax[0].plot(h_seed,mean_intra_spl_seed,'*',color='red',label='Seed graph')
    ax[0].legend()
    ax[1].set_title('GNN Classification Test accuracy')
    ax[1].set_xlabel(f'Homophily Level (p={p})')
    ax[1].set_ylabel('GNN Classification Test accuracy')

    ax[1].plot(h_list,acc_list,'.')
    # add the seed graph to the plot, with a different marker in red
    ax[1].plot(h_seed,acc_seed,'*',color='red',label='Seed graph')
    
    # plot the mean degree with respect to homophily
    ax[2].set_title('Mean degree')
    ax[2].set_xlabel(f'Homophily Level (p={p})')
    ax[2].set_ylabel('Mean degree')
    mean_degree_list = [g.in_degrees().float().mean() for g in g_list]
    ax[2].plot(h_list,mean_degree_list,'.')
    # add the seed graph to the plot, with a different marker in red
    ax[2].plot(h_seed,g_seed.in_degrees().float().mean(),'*',color='red',label='Seed graph')


    plt.show()


def plot_intra_full_spl_ratio(g_list,g_seed,mean_spl_list,mean_intra_spl_list,p=1):
  """
  Plot the ratio of the mean shortest path lengths between the same classes and the mean
  shortest path lengths between all nodes as a function of homophily 

  Args:
      g_list: list of dgl graphs
      g_seed: seed graph
      mean_spl_list: list of mean shortest path lengths of the graphs in g_list
      mean_intra_spl_list: list of mean shortest path lengths between nodes of the same class of the graphs in g_list
      p: homophily level
  Returns:
      None
  
  """
  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111)
  ax.grid(color='gray', linestyle='-', linewidth=0.1)
  ax.set_xlabel(f'Homophily Level (p={p})')
  ax.set_ylabel('Mean intra shortest path / Mean shortest path')
  h_list = [class_homophily(g,p=p).item() for g in g_list]
  mean_spl_arr = np.array(mean_spl_list)
  mean_intra_spl_arr = np.array(mean_intra_spl_list)
  ax.plot(h_list,mean_intra_spl_arr/mean_spl_arr,'.',label='Mean intra shortest path / Mean shortest path')
  h_seed = class_homophily(g_seed,p=p)
  mean_spl_seed, mean_intra_spl_seed = calculate_geodesics_homophily([g_seed])
  ax.plot(h_seed,mean_intra_spl_seed[0]/mean_spl_seed[0],'*',color='red',label='Seed graph')
  ax.legend()
  plt.show()
  
  return


def plot_spl_ratio_vs_gnn_acc(g_seed,mean_spl_list,mean_intra_spl_list,acc_list,p=1):
  """
  Plot the ratio of the mean shortest path lengths between the same classes and the mean
  shortest path lengths between all nodes as a function of homophily 

  Args:
      g_list: list of dgl graphs
      g_seed: seed graph
      mean_spl_list: list of mean shortest path lengths of the graphs in g_list
      mean_intra_spl_list: list of mean shortest path lengths between nodes of the same class of the graphs in g_list
      acc_list: list of GNN classification test accuracies of the graphs in g_list
      p: homophily level
  Returns:
      None
  
  """
  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111)
  ax.grid(color='gray', linestyle='-', linewidth=0.1)
  ax.set_xlabel('Mean intra shortest path / Mean shortest path')
  ax.set_ylabel('GNN Classification Test accuracy')

  mean_spl_arr = np.array(mean_spl_list)
  mean_intra_spl_arr = np.array(mean_intra_spl_list)
  ax.plot(mean_intra_spl_arr/mean_spl_arr,acc_list,'.',label='GNN Classification Test accuracy')

  mean_spl_seed, mean_intra_spl_seed = calculate_geodesics_homophily([g_seed])
  acc_seed = test_gnn_vs_homophily([g_seed])
  ax.plot(mean_intra_spl_seed[0]/mean_spl_seed[0],acc_seed,'*',color='red',label='Seed graph')
  ax.legend()
  plt.show()
  
  return  
    

# do the same plot as plot_spl_ratio_vs_gnn_acc but for mutiple sets of graphs

def plot_spl_ratio_vs_gnn_acc_multiple(g_seeds,mean_spl_lists,mean_intra_spl_lists,acc_lists,p=1):
  """
  Plot the ratio of the mean shortest path lengths between the same classes and the mean
  shortest path lengths between all nodes as a function of homophily 

  Args:
      g_seeds: seed graph
      mean_spl_lists: list of mean shortest path lengths of the graphs in g_list
      mean_intra_spl_lists: list of mean shortest path lengths between nodes of the same class of the graphs in g_list
      acc_lists: list of GNN classification test accuracies of the graphs in g_list
      p: homophily level
  Returns:
      None
  
  """
  # graphs side by side in seperate subplots, with one final plot with all the graphs in one plot
  fig, ax = plt.subplots(1,len(g_seeds)+1,figsize=(8*(len(g_seeds)+1),6))
  for i in range(len(g_seeds)):
    ax[i].grid(color='gray', linestyle='-', linewidth=0.1)
    ax[i].set_xlabel('Mean intra shortest path / Mean shortest path')
    ax[i].set_ylabel('GNN Classification Test accuracy')
    ax[i].set_title(f'Graph {i+1}')
    mean_spl_arr = np.array(mean_spl_lists[i])
    mean_intra_spl_arr = np.array(mean_intra_spl_lists[i])
    ax[i].plot(mean_intra_spl_arr/mean_spl_arr,acc_lists[i],'.',label='GNN Classification Test accuracy')

    mean_spl_seed, mean_intra_spl_seed = calculate_geodesics_homophily([g_seeds[i]])
    acc_seed = test_gnn_vs_homophily([g_seeds[i]])
    ax[i].plot(mean_intra_spl_seed[0]/mean_spl_seed[0],acc_seed,'*',color='red',label='Seed graph')
    ax[i].legend()
  ax[-1].grid(color='gray', linestyle='-', linewidth=0.1)
  ax[-1].set_xlabel('Mean intra shortest path / Mean shortest path')
  ax[-1].set_ylabel('GNN Classification Test accuracy')
  ax[-1].set_title('All graphs')
  for i in range(len(g_seeds)):
    mean_spl_arr = np.array(mean_spl_lists[i])
    mean_intra_spl_arr = np.array(mean_intra_spl_lists[i])
    ax[-1].plot(mean_intra_spl_arr/mean_spl_arr,acc_lists[i],'.',label=f'Graph {i+1}')
    # add the seed graph
    mean_spl_seed, mean_intra_spl_seed = calculate_geodesics_homophily([g_seeds[i]])
    acc_seed = test_gnn_vs_homophily([g_seeds[i]])
    ax[-1].plot(mean_intra_spl_seed[0]/mean_spl_seed[0],acc_seed,'*',color='red',label='Seed graph')

  ax[-1].legend()
  plt.show()

  return


# do the same plot as plot_intra_full_spl_ratio but for mutiple sets of graphs

def plot_intra_full_spl_ratio_multiple(g_lists, g_seeds,mean_spl_lists,mean_intra_spl_lists,p=1):
  """
  Plot the ratio of the mean shortest path lengths between the same classes and the mean
  shortest path lengths between all nodes as a function of homophily 

  Args:
      g_list: list of dgl graphs
      g_seeds: seed graph
      mean_spl_lists: list of mean shortest path lengths of the graphs in g_list
      mean_intra_spl_lists: list of mean shortest path lengths between nodes of the same class of the graphs in g_list
      p: homophily level
  Returns:
      None
  
  """
  # graphs side by side in seperate subplots, with one final plot with all the graphs in one plot
  fig, ax = plt.subplots(1,len(g_seeds)+1,figsize=(8*(len(g_lists)+1),6))
  for i in range(len(g_seeds)):
    
    ax[i].grid(color='gray', linestyle='-', linewidth=0.1)
    ax[i].set_xlabel(f'Homophily Level (p={p})')
    ax[i].set_ylabel('Mean intra shortest path / Mean shortest path')
    ax[i].set_title(f'Graph {i+1}')
    h_list = [class_homophily(g,p=p) for g in g_lists[i]]
    mean_spl_arr = np.array(mean_spl_lists[i])
    mean_intra_spl_arr = np.array(mean_intra_spl_lists[i])
    ax[i].plot(h_list,mean_intra_spl_arr/mean_spl_arr,'.',label='Mean intra shortest path / Mean shortest path')
    h_seed = class_homophily(g_seeds[i],p=p)
    mean_spl_seed, mean_intra_spl_seed = calculate_geodesics_homophily([g_seeds[i]])
    ax[i].plot(h_seed,mean_intra_spl_seed[0]/mean_spl_seed[0],'*',color='red',label='Seed graph')
    ax[i].legend()

  # add a final plot with all the graphs in one plot
  ax[-1].grid(color='gray', linestyle='-', linewidth=0.1)
  ax[-1].set_xlabel(f'Homophily Level (p={p})')
  ax[-1].title.set_text('All graphs')
  ax[-1].set_ylabel('Mean intra shortest path / Mean shortest path')
  for i in range(len(g_seeds)):
    h_list = [class_homophily(g,p=p) for g in g_lists[i]]
    mean_spl_arr = np.array(mean_spl_lists[i])
    mean_intra_spl_arr = np.array(mean_intra_spl_lists[i])
    ax[-1].plot(h_list,mean_intra_spl_arr/mean_spl_arr,'.',label=f'Graph {i+1}')
    h_seed = class_homophily(g_seeds[i],p=p)
    mean_spl_seed, mean_intra_spl_seed = calculate_geodesics_homophily([g_seeds[i]])
    ax[-1].plot(h_seed,mean_intra_spl_seed[0]/mean_spl_seed[0],'*',color='red',label='Seed graph')
  plt.show()
  return


def plot_spl_predictions(g,model):
    """
    Plot the distribution of shortest path lengths for correctly and incorrectly predicted nodes

    Args:
        g: dgl graph
        model: trained model
    Returns:
        None
    """
    preds = model(g,g.ndata['feat']).argmax(dim=1)
    labels = g.ndata['label']
    test_mask = g.ndata['test_mask']

    true_pred_mask = (preds==labels) * test_mask
    false_pred_mask = (preds!=labels) * test_mask

    # calculate the shortest path length distribution for correcty and incorrectly predicted nodes
    lengths,freq = shotest_paths_dist(g)
    freq_true, freq_false,freq_pred = np.zeros(lengths.shape), np.zeros(lengths.shape), np.zeros(lengths.shape)

    for c in g.ndata['label'].unique():
        length_pred_c,freq_pred_c = shotest_paths_dist(g,nodes=g.nodes()[(g.ndata['label']==c)*test_mask])
        length_true_c,freq_true_c = shotest_paths_dist(g,nodes=g.nodes()[(g.ndata['label']==c)*true_pred_mask])
        length_false_c,freq_false_c = shotest_paths_dist(g,nodes=g.nodes()[(g.ndata['label']==c)*false_pred_mask])
        
        true_dist_c = dict(zip(length_true_c,freq_true_c))
        false_dist_c = dict(zip(length_false_c,freq_false_c))
        pred_dist_c = dict(zip(length_pred_c,freq_pred_c))

        for l in lengths:
            if l in true_dist_c:
                freq_true[l] += true_dist_c[l]
            if l in false_dist_c:
                freq_false[l] += false_dist_c[l]
            if l in pred_dist_c:
                freq_pred[l] += pred_dist_c[l]
                

    freq_true = np.array(freq_true)/np.sum(freq_true)
    freq_false = np.array(freq_false)/np.sum(freq_false)
    freq_pred = np.array(freq_pred)/np.sum(freq_pred)

    plt.figure(1,figsize=(8,6))
    plt.plot(lengths,freq_true,label='Correctly predicted')
    plt.plot(lengths,freq_false,label='Incorrectly predicted')
    plt.plot(lengths,freq_pred,label='Predicted')

    # add dashed line for mean shortest path length of incorrect nodes and correct nodes
    mean_spl_false = np.sum(lengths*freq_false)
    mean_spl_true = np.sum(lengths*freq_true)
    mean_spl_pred = np.sum(lengths*freq_pred)

    std_spl_false = np.sqrt(np.sum((lengths-mean_spl_false)**2*freq_false))
    std_spl_true = np.sqrt(np.sum((lengths-mean_spl_true)**2*freq_true))
    std_spl_pred = np.sqrt(np.sum((lengths-mean_spl_pred)**2*freq_pred))

    plt.axvline(x=mean_spl_false,linestyle='--',color='r',label='Mean spl of incorrectly predicted nodes')
    plt.axvline(x=mean_spl_true,linestyle='--',color='g',label='Mean spl of correctly predicted nodes')
    plt.axvline(x=mean_spl_pred,linestyle='--',color='b',label='Mean spl of predicted nodes')

    # add a faint window showing the standard deviation of the mean shortest path length
    plt.axvspan(mean_spl_false-std_spl_false,mean_spl_false+std_spl_false,color='r',alpha=0.1)
    plt.axvspan(mean_spl_true-std_spl_true,mean_spl_true+std_spl_true,color='g',alpha=0.1)
    plt.axvspan(mean_spl_pred-std_spl_pred,mean_spl_pred+std_spl_pred,color='b',alpha=0.2)

    #plt.plot(lengths,freq_full,label='full')
    plt.legend()
    plt.show()

    return


# compare p-degree homophily of different types for different graphs, ovee p - the size of the neighbourhood to consider.
# for each graph in g_list, plot the p-degree homophily over p on the the same plot
# do seperate plots for different variants of homohpliy
def plot_p_degree_homophily(g_list, homophily_type='edge',class_normalised=False):
    """
    Plot the p-degree homophily of the graphs in g_list as a function of p - the size of the neighbourhood to consider
    
    Args:
        g_list: list of dgl graphs
        homophily_type: type of homophily to calculate, one of 'edge', 'node'
        class_normalised: whether to normalise the class distribution of the neighbourhoods before calculating homophily
    Returns:
        None
    
    """
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.grid(color='gray', linestyle='-', linewidth=0.1)
    ax.set_xlabel('p')
    ax.set_ylabel('p-degree homophily')
    ax.set_title(f'{homophily_type} homophily')
    for i,g in enumerate(g_list):
        if homophily_type == 'edge':
            h_list = [edge_homophily(g,p=p,class_normalised=class_normalised) for p in range(1,10)]
        elif homophily_type == 'node':
            h_list = [node_homophily(g,p=p,class_normalised=class_normalised) for p in range(1,10)]
        ax.plot(range(1,10),h_list,label=f'Graph {i+1}')
    ax.legend()
    plt.show()
    return




 