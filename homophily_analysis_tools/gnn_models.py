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