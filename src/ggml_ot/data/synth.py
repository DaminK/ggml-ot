from .util import create_t_triplets
from torch.utils.data import Dataset
import pandas as pd

from ggml.distances import pairwise_mahalanobis_distance_npy
import ot
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from ggml.plot import plot_emb, hier_clustering

class synthetic_Dataset(Dataset):
    #The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, *args, **kwargs):
        #Generate syntehtic data
        distributions, distributions_labels, points, point_labels, distribution_modes = get_pointcloud(*args, **kwargs)

        #Population-level 
        self.distributions = distributions
        self.distributions_labels = distributions_labels
        self.distribution_modes = distribution_modes

        #Unit-level 
        self.datapoints = points 
        self.datapoints_labels = point_labels

        #Triplets
        self.triplets = create_t_triplets(distributions,distributions_labels,**kwargs)

    def __len__(self):
        #Datapoints to train are always given as triplets
        return len(self.triplets)

    def __getitem__(self, idx):
        #Returns elements and labels of triplet at idx
        i,j,k = self.triplets[idx]
        return np.stack((self.distributions[i],self.distributions[j],self.distributions[k]),dtype="f"),np.stack((self.distributions_labels[i],self.distributions_labels[j],self.distributions_labels[k]),dtype="f")
    
    def get_raw_distributions(self):
        return self.distributions,self.distributions_labels
    
    def compute_OT_on_dists(self,ground_metric = None,w = None,legend="Side"):
        D = np.zeros((len(self.distributions),len(self.distributions)))
        for i,distribution_i in enumerate(self.distributions):
            for j,distribution_j in enumerate(self.distributions):
                if i < j:
                    if w is not None:
                        M = pairwise_mahalanobis_distance_npy(distribution_i,distribution_j,w)
                        D[i,j] = ot.emd2([],[],M)
                else:
                    D[i,j]=D[j,i]
        
        hardcoded_symbols = [i % 10 for i in range(len(self.distributions))]
        plot_emb(D,method='umap',colors=self.distributions_labels,symbols=hardcoded_symbols,legend=legend,title="UMAP",verbose=True,cmap=sns.cubehelix_palette(as_cmap=True),annotation=None,s=200)

        hier_clustering(D,self.distributions_labels, ax=None,cmap=sns.cubehelix_palette(as_cmap=False,n_colors=len(np.unique(self.distributions_labels))),dist_name="W_θ")
        return D
    

def get_pointcloud(distribution_size=100, class_means = [0,5,10], offsets = [0,5,10,15], shared_means_x = [], shared_means_y = [], plot=True, varying_size=True,noise_scale=1000,noise_dims=1,return_dict=False,*args, **kwargs):

    #Gaussian along dim 1, uniform along dim 2 (only information is the mean of the gaussian)
    unique_label = np.arange(len(class_means),dtype=int)

    distributions = []
    distributions_labels = []
    plotting_df =[]

    label_distribution_modes = []

    for mean,label in zip(class_means,unique_label):
        i = 0
        for offset in offsets:
            rand_size= np.random.randint(20,distribution_size) if varying_size else distribution_size

            dim1 = np.random.normal(10+mean,size=rand_size,scale=1.5)
            dim2 = np.random.uniform(7.5+offset,12.5+offset,size=(rand_size,noise_dims))

            label_distribution_modes = label_distribution_modes + [1]*rand_size

            for shared_mean_x,shared_mean_y in zip(shared_means_x,shared_means_y):
                dim1 = np.concatenate((dim1,np.random.normal(shared_mean_x,size=rand_size,scale=1.5)))
                dim2 = np.concatenate((dim2,np.random.normal(shared_mean_y,size=(rand_size,noise_dims),scale=1.5)),axis=0) # #np.random.normal(2.5+offset,size=n)
                label_distribution_modes = label_distribution_modes + [0]*rand_size

            dim1 = dim1 * 5 / 4
            dim2 = dim2*noise_scale

            stacked = np.insert(dim2,0,dim1,axis=1)

            #stacked = np.append(dim2,[dim1],axis=0)
            #stacked = np.stack((dim1,dim2),axis=-1)
            plotting_df.append(pd.DataFrame({'x':dim1,'y':dim2[:,0],'class':label,'distribution':i}))

            distributions.append(stacked)
            distributions_labels.append(label)
        
            i+=1


    if plot:
        df = pd.concat(plotting_df, axis=0)

        plt.figure(figsize=(6,5))
        ax = sns.scatterplot(df,x='x',y='y',hue="class",style='distribution')
        sns.move_legend(ax, "center right", bbox_to_anchor=(1.3, 0.5))

        #xticks = ax.xaxis.get_major_ticks()
        #xticks[0].label1.set_visible(False)
        #yticks = ax.yaxis.get_major_ticks()
        #xticks[-1].label1.set_visible(False)

        plt.show()


    points = np.concatenate(distributions) #np.reshape(np.asarray(dists),(-1,2))
    point_labels = sum([[l] * len(D) for l,D in zip(distributions_labels,distributions)],[]) #flattens list of lists

    if return_dict:
        data_dict = {}
        data_dict["distributions"],data_dict["distributions_labels"],data_dict["points"], data_dict["point_labels"], data_dict["patient"] = distributions, distributions_labels, points, point_labels, label_distribution_modes
        return data_dict
    else:
        return distributions, distributions_labels, points, point_labels, label_distribution_modes