import torch
from torch_geometric.utils import to_networkx, degree
from dataset import HW3Dataset
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]

    res = data.x.reshape(-1, 1).squeeze(1).cpu()
    bin_edges = np.linspace(res.min(), res.max(), 150)
    fig = plt.figure(figsize=(12, 8))
    sns.histplot(res[data.train_mask], kde=True, bins=bin_edges, stat="density", label="train", alpha=0.5)
    sns.histplot(res[data.val_mask], kde=True, bins=bin_edges, stat="density", label="val", alpha=0.5)
    plt.xlabel("Feature Value")
    plt.title("Distribution of Feature Values")
    plt.legend()
    plt.savefig("feature_dist.png")

    res = data.node_year.squeeze(1).cpu()
    bin_edges = np.arange(min(res), max(res)+2) - 0.5

    fig = plt.figure(figsize=(12, 8))
    sns.histplot(res[data.train_mask], kde=True, bins=bin_edges, stat="density", label="train", alpha=0.5)
    sns.histplot(res[data.val_mask], kde=True, bins=bin_edges, stat="density", label="val", alpha=0.5)
    plt.xlabel("Year")
    plt.title("Distribution of Years")
    plt.legend()
    plt.savefig("year_dist.png")

    res = data.y.squeeze(1).cpu()
    bin_edges = np.arange(min(res), max(res)+2) - 0.5

    fig = plt.figure(figsize=(12, 8))
    sns.histplot(res[data.train_mask], kde=True, bins=bin_edges, stat="density", label="train", alpha=0.5)
    sns.histplot(res[data.val_mask], kde=True, bins=bin_edges, stat="density", label="val", alpha=0.5)
    plt.xlabel("Class")
    plt.title("Distribution of Class Labels")
    plt.legend()
    plt.savefig("class_dist.png")