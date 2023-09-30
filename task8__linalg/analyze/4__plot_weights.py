import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

def extract_weights(module):
    x = torch.tensor([])
    for p in module.parameters(): x = torch.cat((x, p.detach().flatten()))
    return x

def plot_weights(weights):
    n = int(np.ceil(np.sqrt(weights.shape[0])))
    weights = torch.cat((weights, torch.zeros(n**2 - weights.shape[0])))
    weights = weights.reshape(n, n)
    sns.heatmap(weights, cmap='gray')
    plt.show()

def main(module):
    weights = extract_weights(module)
    plot_weights(weights)

