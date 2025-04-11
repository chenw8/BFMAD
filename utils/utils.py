import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from kmeans_pytorch import kmeans
import time
from sklearn.mixture import GaussianMixture

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def gmm_clustering(x, n_components, d_model):
    start = time.time()
    x_np = x.cpu().detach().numpy()
    x_np = x_np.reshape((-1, d_model))
    
    print('running Gaussian Mixture Model Clustering. It takes few minutes to find clusters')
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=0)
    gmm.fit(x_np)
    
    print("time for conducting GMM Clustering:", time.time() - start)
    print('GMM clustering is done!!!')
    
    cluster_centers = torch.from_numpy(gmm.means_).float()
    
    return to_var(cluster_centers) 

def k_means_clustering(x,n_mem,d_model):
    start = time.time()
    x = x.contiguous() 
    x = x.view([-1,d_model])
    print('running K Means Clustering. It takes few minutes to find clusters')
    # sckit-learn xxxx (cuda problem)
    _, cluster_centers = kmeans(X=x, num_clusters=n_mem, distance='euclidean', device=torch.device('cuda:0'))
    print("time for conducting Kmeans Clustering :", time.time() - start)
    print('K means clustering is done!!!')

    return cluster_centers