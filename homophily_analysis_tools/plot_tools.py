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

from homophily_analysis_tools.spl_tools import calculate_geodesics_homophily, test_gnn_vs_homophily,shotest_paths_dist
from homophily_analysis_tools.homophily_metrics import homophily


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
    h_list.append(homophily(g,p=p))
  
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
    h_list.append(homophily(g, p=p))
  
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
    h_list.append(homophily(g,p=p))
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
    plt.hist(d_list,bins=100,alpha=alpha.item(),label='h = '+str(round(homophily(g).item(),2)),density=True)

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
    h_list = [homophily(g).item() for g in g_list]
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
    h_list = [homophily(g,p=p) for g in g_list]

    h_seed = homophily(g_seed,p=p)
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
  h_list = [homophily(g,p=p).item() for g in g_list]
  mean_spl_arr = np.array(mean_spl_list)
  mean_intra_spl_arr = np.array(mean_intra_spl_list)
  ax.plot(h_list,mean_intra_spl_arr/mean_spl_arr,'.',label='Mean intra shortest path / Mean shortest path')
  h_seed = homophily(g_seed,p=p)
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
    h_list = [homophily(g,p=p) for g in g_lists[i]]
    mean_spl_arr = np.array(mean_spl_lists[i])
    mean_intra_spl_arr = np.array(mean_intra_spl_lists[i])
    ax[i].plot(h_list,mean_intra_spl_arr/mean_spl_arr,'.',label='Mean intra shortest path / Mean shortest path')
    h_seed = homophily(g_seeds[i],p=p)
    mean_spl_seed, mean_intra_spl_seed = calculate_geodesics_homophily([g_seeds[i]])
    ax[i].plot(h_seed,mean_intra_spl_seed[0]/mean_spl_seed[0],'*',color='red',label='Seed graph')
    ax[i].legend()

  # add a final plot with all the graphs in one plot
  ax[-1].grid(color='gray', linestyle='-', linewidth=0.1)
  ax[-1].set_xlabel(f'Homophily Level (p={p})')
  ax[-1].title.set_text('All graphs')
  ax[-1].set_ylabel('Mean intra shortest path / Mean shortest path')
  for i in range(len(g_seeds)):
    h_list = [homophily(g,p=p) for g in g_lists[i]]
    mean_spl_arr = np.array(mean_spl_lists[i])
    mean_intra_spl_arr = np.array(mean_intra_spl_lists[i])
    ax[-1].plot(h_list,mean_intra_spl_arr/mean_spl_arr,'.',label=f'Graph {i+1}')
    h_seed = homophily(g_seeds[i],p=p)
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


# compare p-level homophily of different types for different graphs, ovee p - the size of the neighbourhood to consider.
# for each graph in g_list, plot the p-level homophily over p on the the same plot
# do seperate plots for different variants of homohpliy
def plot_p_level_homophily(g_list, node_h_type = 'symmetric', homophily_type='edge',class_normalised=False):
    """
    Plot the p-level homophily of the graphs in g_list as a function of p - the size of the neighbourhood to consider
    
    Args:
        g_list: list of dgl graphs
        node_h_type: type of node homophily to calculate, one of 'symmetric', 'random walk'
        homophily_type: type of homophily to calculate, one of 'edge', 'node'
        class_normalised: whether to normalise the class distribution of the neighbourhoods before calculating homophily
    Returns:
        None
    
    """
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.grid(color='gray', linestyle='-', linewidth=0.1)
    ax.set_xlabel('p')
    ax.set_ylabel('p-level homophily')
    ax.set_title(f'{"class normalised"*class_normalised} {homophily_type} homophily')
    for i,g in enumerate(g_list):
        if homophily_type == 'edge':
            h_list = [homophily(g,p=p,type='edge',class_normalised=class_normalised) for p in range(1,10)]
        elif homophily_type == 'random walk':
            h_list = [homophily(g,p=p,type='rw',class_normalised=class_normalised) for p in range(1,10)]
        elif homophily_type == 'symmetric':
            h_list = [homophily(g,p=p,type='sym',class_normalised=class_normalised) for p in range(1,10)]
        ax.plot(range(1,10),h_list,label=f'Graph {i+1}')
    ax.legend()
    plt.show()
    return




 