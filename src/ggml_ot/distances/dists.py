import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy
from ggml_ot.plot import plot_ellipses, plot_clustermap, plot_emb
import ot

#wrapper for precomputed distance matrix
#only execute once values are accessed
class Computed_Distances():
    def __init__(self, points, theta, n_threads=60):
        self.n_treads = n_threads
        self.points = points
        self.theta = theta

        
        self.data = np.full((len(points),len(points)), np.nan)

        self.ndim = self.data.ndim
        self.sape = self.data.shape


    

    def __getitem__(self, slice_):

        if np.isnan(self.data[slice_]).any():
            ranges = [np.squeeze(np.arange(len(self.data))[slice_[i]]) for i in range(len(slice_))] 
            entry_nan_index = ([],[])
            for entry in ranges[0]:
                #print(self.data[entry,:].ndim)
                check = np.isnan(self.data[entry,:])
                if check.ndim == 2 and np.isnan(self.data[entry,:][:,slice_[1]]).any():
                    entry_nan_index[0].append(entry) 
                elif check.ndim == 1 and np.isnan(self.data[entry,:][slice_[1]]).any():
                    entry_nan_index[0].append(entry)   
            for entry in ranges[1]:
                if np.isnan(self.data[slice_[0],entry]).any():
                    entry_nan_index[1].append(entry)   
            
            #check for elements with nan entries
            dist = pairwise_mahalanobis_distance_npy(self.points[entry_nan_index[0],:],self.points[entry_nan_index[1],:],w=self.theta, numThreads = n_threads)
            self.data[np.ix_(entry_nan_index[0],entry_nan_index[1])] = dist 

            return self.data[slice_]

        else:
            return self.data[slice_]
        

def compute_OT(distributions,labels,precomputed_distances=None,ground_metric = None,w = None,legend=None,numThreads=32):
    D = np.zeros((len(distributions),len(distributions)))
    for i,distribution_i in enumerate(distributions):
        for j,distribution_j in enumerate(distributions):
            if i < j:
                if precomputed_distances is not None:
                    start_i = int(np.sum([len(dist) for dist in distributions[:i]]))
                    start_j = int(np.sum([len(dist) for dist in distributions[:j]]))
                    if precomputed_distances.ndim == 1:
                        precomputed_distances = scipy.spatial.distance.squareform(precomputed_distances)
                    M = precomputed_distances[start_i:start_i+len(distribution_i),start_j:start_j+len(distribution_j)]
                elif w is not None:
                    M = pairwise_mahalanobis_distance_npy(distribution_i,distribution_j,w)

                D[i,j] = ot.emd2([],[],M,numThreads=numThreads)
                #TODO handle non mahalanobis distances
            else:
                D[i,j]=D[j,i]
    
    hardcoded_symbols = None #[i % 4 for i in range(len(distributions))]
    plot_emb(D,method='umap',colors=labels,symbols=hardcoded_symbols,legend=legend,title="UMAP",verbose=True,annotation=None,s=200)
    # plot_emb(D,method='diffusion',colors=labels,symbols=hardcoded_symbols,legend=legend,title="DiffMap",verbose=True,annotation=None,s=200)

    plot_clustermap(D,labels,dist_name="W_θ")
    return D


def pairwise_mahalanobis_distance(X_i,X_j,w):
    # W has shape (rank k<=dim) x dim
    # X_i, X_y have shape n x dim, m x dim
    # return Mahalanobis distance between pairs n x m 

    #Transform poins of X_i,X_j according to W
    if w.dim() == 1:
        #assume cov=0, scale dims by diagonal
        proj_X_i = X_i * w[None,:]
        proj_X_j = X_j * w[None,:]

    else: 
        w = torch.transpose(w,0,1)
        proj_X_i = torch.matmul(X_i,w)
        proj_X_j = torch.matmul(X_j,w)

    return torch.linalg.norm(proj_X_i[:,torch.newaxis,:]  -  proj_X_j[torch.newaxis,:,:],dim=-1)    

'''
def pairwise_mahalanobis_distance_npy(X_i,X_j,w=None):
    # W has shape dim x dim
    # X_i, X_y have shape n x dim, m x dim
    # return Mahalanobis distance between pairs n x m 
    if w is None:
        w = np.identity(X_i.shape[-1])
    else:
        w = w.astype("f")

    X_i = X_i.astype("f")
    X_j = X_j.astype("f")

    #Transform poins of X_i,X_j according to W
    if w.ndim == 1:
        #assume cov=0, scale dims by diagonal
        #w = np.diag(w)
        #proj_X_i = np.matmul(X_i,w)
        #proj_X_j = np.matmul(X_j,w)

        proj_X_i = X_i * w[None,:]
        proj_X_j = X_j * w[None,:]

    else: 
        w = np.transpose(w)
        proj_X_i = np.matmul(X_i,w)
        proj_X_j = np.matmul(X_j,w)

    return np.linalg.norm(proj_X_i[:,np.newaxis,:]  -  proj_X_j[np.newaxis,:,:],axis=-1)  '
'''

def pairwise_mahalanobis_distance_npy(X_i,X_j=None,w=None,numThreads=32):
    # W has shape dim x dim
    # X_i, X_y have shape n x dim, m x dim
    # return Mahalanobis distance between pairs n x m 
    if X_j is None:
        if w is None or isinstance(w,str):
            return pairwise_distances(X_i,metric=w,n_jobs=numThreads) #cdist .. ,X_j)
        else:
            if w.ndim == 2 and w.shape[0]==w.shape[1]:
                return pairwise_distances(X_i,metric="mahalanobis",n_jobs=numThreads,VI =w)    
            else:
                X_j = X_i
    #Transform poins of X_i,X_j according to W
    if w is None or isinstance(w,str):
        return scipy.spatial.distance.cdist(X_i,X_j,metric=w)
    #Assume w is cov matrix of mahalanobis distance
    elif w.ndim == 1:
        #assume cov=0, scale dims by diagonal
        w = np.diag(w)
        proj_X_i = np.matmul(X_i,w)
        proj_X_j = np.matmul(X_j,w)

        #proj_X_i = X_i * w[None,:]
        #proj_X_j = X_j * w[None,:]
    else: 
        w = np.transpose(w)
        proj_X_i = np.matmul(X_i,w)
        proj_X_j = np.matmul(X_j,w)
    
    return np.linalg.norm(proj_X_i[:,np.newaxis,:]  -  proj_X_j[np.newaxis,:,:],axis=-1)  

def plot_w_theta(w_theta=None,M=None,ax=None):
    if M is None:
        if isinstance(w_theta, torch.Tensor):
            W = w_theta.clone().detach().numpy()
        else:
            W = w_theta
        M = np.dot(W,np.transpose(W))
    M = M / np.linalg.norm(M)
    return plot_ellipses(M,ax=ax)