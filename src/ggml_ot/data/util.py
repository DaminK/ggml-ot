import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def create_triplets(distributions,labels):
    triplets = []
    for i,_ in enumerate(distributions):
        for j,_ in enumerate(distributions):
            for k,_ in enumerate(distributions):
                if labels[i]==labels[j] and labels[j] != labels[k] and i != j:
                    triplets.append((i,j,k))
    return triplets

def create_t_triplets(distributions,labels,t=5,**kwargs):
    print(f"passed neighs: {t}")
    labels= np.asarray(labels)
    triplets = []
    replace = any(np.unique(labels,return_counts=True)[1]<t)
    

    def get_neighbors(class_,skip=None):
        #get t elements from distributions where labels = class
        #TODO optional skip self
        return np.random.choice(np.where(labels == class_)[0],size=t,replace= replace)
    
    for j,_ in enumerate(distributions):

        c_j = labels[j]
        for i in get_neighbors(c_j):
            for c_k in np.unique(labels):
                if c_k != c_j:
                    for k in get_neighbors(c_k):
                        triplets.append((i,j,k))
    return triplets

