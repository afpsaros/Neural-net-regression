# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:50:09 2020

@author: afpsaros
"""


import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

X, y = load_digits(return_X_y=True)

print(len(X))

print(len(X[0]))
#%%
tsne = TSNE()
X_embedded = tsne.fit_transform(X)
#%%
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue = y, legend='full', palette=palette)