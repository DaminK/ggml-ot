import numpy as np
import scipy

import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib as mpl

import umap
import pandas as pd
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn.decomposition import PCA
from pydiffmap import diffusion_map

import scipy.spatial as sp, scipy.cluster.hierarchy as hc
#import phate
import warnings

#import MDAnalysis.analysis.diffusionmap as diffusionmap

def plot_distribution(dists,labels,projection= lambda x:x,title="",legend=True):
    pca = None
    dim = dists.shape[-1]
    if dim>2:
        print("PCA")
        print(dists.shape)
        flat_dists = projection(dists.reshape(-1,dim))
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit_transform(flat_dists)

    i=0
    dfList_projected = []
    for dist, l in zip(dists,labels):
        stacked_projected = projection(dist) 
        if pca is not None:
            stacked_projected = pca.transform(stacked_projected)
        dfList_projected.append(pd.DataFrame({'x':stacked_projected[:,0],'y':stacked_projected[:,1],'class':str(l.item()),'dist':i % 4})) #TODO ahrdcoded
        i += 1

    df_projected= pd.concat(dfList_projected, axis=0)
    plt.figure(figsize=(6,6))
    ax = sns.scatterplot(df_projected,x='x',y='y',hue="class",style='dist',alpha=0.5)
    
    #ax.set_ylim(-0.01, 0.01)
    if legend:
        sns.move_legend(ax, "center right", bbox_to_anchor=(1.3, 0.5))
    else:
        ax.get_legend().remove()

    ax.set_title(title)

def plot_ellipses(covariances, ax=None, title=None):
        if ax is None:
            print("Create fig")
            fig, ax = plt.subplots(figsize=(3,3))
           
            
        max = 0

        covariances = covariances / np.max(covariances)

            
        n_classes = 1 #for now just 1    
        colors = sns.color_palette("Set2",n_classes)
        for n, color in enumerate(colors):
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mpl.patches.Ellipse(
                [0]*len(covariances), v[0], v[1], angle=180 + angle, color=color
            )
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            ax.set_aspect("equal", "datalim")
            max = np.max([max,v[0],v[1]])

        max  = 1.8 # 1.5* np.max(covariances)
    
        ax.set_xlim([-max, max])
        ax.set_ylim([-max, max])
        
        plt.xticks([-1,0,1]) #(np.arange(-max,max,dtype=int)))
        plt.yticks([-1,0,1]) #(np.arange(-max,max,dtype=int)))
        


        if title is not None:
            ax.set_title(title)

        return ax

def plot_heatmap(results,labels='auto',xlabels=None,ylabels=None,ax=None,title=None):
    xlabels = labels if xlabels is None else xlabels
    ylabels = labels if ylabels is None else ylabels
    ax = sns.heatmap(results,xticklabels=xlabels, yticklabels=ylabels,ax=ax, square= results.shape[0] == results.shape[1])
    ax.set_title(f'pairwise distances' if title is None else title)

def plot_transport_plan(plan,ax=None):
    plt.figure()
    sns.heatmap(plan)
    plt.show()

def plot_emb(dists,method='umap',precomputed_emb=None,colors=None,symbols=None,ax=None, Cluster_ID=None,title=None,cmap="tab20",save_path=None,verbose=True,legend='Top',s=15,hue_order=None,annotation=None, linewidth=0.02,annotation_image_path=None):
    if precomputed_emb is None:
        if method == 'umap':
            with warnings.catch_warnings():
                #unfortunatly UMAP throws a warning that transformations of data points is not possible using precomputed metrics. We do not think that this is a relveant information to users and the warning can not be masked through parameters, so we catch it manually here.
                warnings.simplefilter("ignore",category=UserWarning)
                reducer = umap.UMAP(metric='precomputed') #,n_neighbors=10
                emb = reducer.fit_transform(dists)
        elif method == 'tsne':
            emb = TSNE(n_components=2, metric='precomputed', learning_rate='auto', init='random', perplexity=3).fit_transform(dists)
        elif method == "diffusion":
            mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = 2, epsilon =0.1, alpha = 0.5, k=64)

            emb = mydmap.fit_transform(dists /dists.max())
            emb=emb[:,[0,1]]

        elif method == "fast_diffusion":
            epsilon = 0.1
            scaled_matrix = dists ** 2 / epsilon

            # take negative exponent of scaled matrix to create Isotropic kernel
            kernel = np.exp(scaled_matrix)
            D_inv = np.diag(1 / kernel.sum(1))
            diff = np.dot(D_inv, kernel)
            eigenvals, eigenvectors = np.linalg.eig(diff)
            print(eigenvals)
            sort_idx = np.argsort(eigenvals)[::-1]
            #eigenvalues = eigenvals[sort_idx]
            eigenvectors = eigenvectors[sort_idx]
            print(eigenvectors.shape)
            emb = np.transpose(eigenvectors[[0,1],:])
            #emb=emb[:,[0,1]]

        elif method == "mds":
            mds = manifold.MDS(n_components=2, dissimilarity="precomputed",normalized_stress='auto')
            emb = mds.fit_transform(dists)
        elif method == "phate":
            phate_op = phate.PHATE(knn_dist="precomputed_distance",verbose=0)
            emb = phate_op.fit_transform(dists,)       
    else:
        emb = precomputed_emb
        #emb = results.embedding_ 

    
    df_embed = pd.DataFrame(emb,columns=['x','y'])
    
    df_embed['Classes']=colors
    df_embed['Condition']=symbols
    df_embed['annotation'] = annotation

    #size = None if Cluster_ID is None else [6 if is_cluster else 1 for is_cluster in Cluster_ID]
    #df_embed['size'] = size
    df_embed['Type'] =  None if Cluster_ID is None else ["Cluster" if is_cluster else "Trial" for is_cluster in Cluster_ID]
    type_to_size = {
        "Cluster": 50,
        "Trial": 7,
        None: 3 if annotation_image_path is None else 200
    }
    
    if ax is None:
        fig = plt.figure(figsize=(5,5)) if annotation_image_path is None else plt.figure(figsize=(30,7))
        #ax = fig.add_subplot(111, aspect='equal')

    ax = sns.scatterplot(df_embed, x='x',y='y',edgecolor="white",alpha=1.0,s=s,linewidth=linewidth,hue='Classes' if colors is not None else None, style='Condition' if symbols is not None else None, size="Type" if Cluster_ID is not None else None,sizes=type_to_size if Cluster_ID is not None else None, ax=ax,palette=cmap, legend = False if not legend else 'auto',hue_order=hue_order)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


    
    #plt.show()
    plt.subplots_adjust(top=10/12)

    if verbose:
        ax.set_title(f'{method} projection' if title is None else title) #plt.gca().set_aspect('equal', 'datalim')

    if legend=='Top':
        ax.legend(loc='upper center', bbox_to_anchor=(0., 1.075,0.9, .10) if len(np.unique(colors))>5 else (0., 1.05,0.9, .075),  prop=dict(weight='bold'),handletextpad=0.1,frameon=False, shadow=False, ncol=4 if len(np.unique(colors))>5 else 5 ,mode="expand")
    elif legend=='Side':
        plt.legend(frameon=False)
        sns.move_legend(ax, "right", bbox_to_anchor=(1.5, 0.5))
        



    #TODO move
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image

    def crop(im, w, h):
        width, height = im.size # Get dimensions
        left = (1-w)/2*width
        top = (1-h)/2*height
        right = (w+1)/2*width
        bottom = (h+1)/2*height
        return im.crop((left, top, right, bottom))

    def getImage(path, zoom=0.5,w=0.6,h=0.72): #0.04
        return OffsetImage(np.asarray(crop(Image.open(path),w,h)), zoom=zoom)
    
    

    if annotation_image_path is not None:
        if "histo" in annotation_image_path[0]:
            scaling = 0.025
            width = 0.8
            height = 0.8
        elif "niche" in annotation_image_path[0]:
            scaling = 0.45
            width = 0.6
            height = 0.75
        else:
            #Probably cell distr
            scaling = 0.4
            width = 0.8
            height = 0.8
            #print("unknown images, no scaling")

        for p in range(0,df_embed.shape[0]):
            ab = AnnotationBbox(getImage(annotation_image_path[p],zoom=scaling,w=width,h=height), 
                                (df_embed.x[p], df_embed.y[p]), 
                                xycoords='data',
                                boxcoords="offset points",
                                frameon=False,
                                box_alignment=(0,0),pad=0.1)
            ax.add_artist(ab)

    # add annotations one by one with a loop, credit https://python-graph-gallery.com/46-add-text-annotation-on-scatterplot/
    if annotation is not None:
        for p in range(0,df_embed.shape[0]):
            ax.text(df_embed.x[p], df_embed.y[p], df_embed.annotation[p], horizontalalignment='left', size='x-small', color='black') #, weight='semibold')

    if save_path is not None:
        print(save_path)
        ax.figure.savefig(save_path)

    return emb

def plot_clustermap(dist,labels, ax=None, cluster=True, method="average",title=None,dist_name="",log=False,save_path=None,cmap="tab20",hue_order=None,annotation=False):
    #Creating list of colors for conds
    unique_inds = np.unique(labels, return_index=True)[1]
    unique_labels = np.asarray([labels[index] for index in sorted(unique_inds)] ).tolist() if hue_order is None else hue_order
    if isinstance(cmap,str):
        cmap = sns.color_palette(palette=cmap,n_colors=len(unique_labels))
        colors = [cmap[unique_labels.index(l)] for l in labels]
    else:
        colors = [cmap[l] for l in labels]

    dist = np.copy(dist)

    if cluster:
        dist[np.eye(len(dist),dtype=bool)] = 0  
        linkage =  hc.linkage(sp.distance.squareform(dist), method=method, optimal_ordering=True)
    else:
        linkage = None

    norm = None
    if log:
        norm = LogNorm()
        dist[dist<=0] = np.min(dist[dist>0])
        #dist[dist>0] = np.log(dist[dist>0])
        
        #dist = np.log2(dist)
        
        #dist = np.nan_to_num(dist, nan=np.nanmin(dist))
        #dist[dist==0] = 50 #np.amin(dist)
        #dist_name = "log " + dist_name
    #dist=squareform(dist)

    
    # convert the redundant n*n square matrix form into a condensed nC2 array
    #print(dist.shape)
    #dist = ssd.squareform(dist) 
    #print(dist.shape)
    
    #

    
    fig = sns.clustermap(dist, figsize=(5,5),row_cluster=cluster, col_cluster=cluster, row_linkage=linkage ,col_linkage=linkage, dendrogram_ratio=0.15,
            row_colors=colors, col_colors=colors,method=method, cmap=sns.cm.rocket_r,
            #cbar_pos=(.2, 0, .2, .02),cbar_kws={"label":dist_name, "orientation":'horizontal'},yticklabels=False,xticklabels=False) #,)
            cbar_pos=(0.05, 0.1, .1, .02),cbar_kws={"orientation":'horizontal'},yticklabels=False,xticklabels=annotation,norm=norm)
    #fig.ax_heatmap.set(xlabel = "Graphs",ylabel ="Graphs")

    fig.ax_heatmap.tick_params(right=False, bottom=False if annotation is False else True) #sounds really stupid but annotation might be non-boolean


    fig.ax_col_dendrogram.set_visible(False)

    fig.ax_cbar.set_title(dist_name)

    #if cluster:
    #    fig.ax_heatmap.set_title(f'{method}')
    

    #handles = [Patch(facecolor=cmap[unique_labels.tolist().index(l)]) for l in unique_labels]
    #plt.legend(handles, unique_labels, title='Classes',frameon=False,
    #       bbox_to_anchor=(-0.03, 0.01), bbox_transform=plt.gcf().transFigure, loc='lower left')
    
    #h = [plt.plot([],[], color=cmap[c], marker="s", ms=c, ls="")[0] for c in range(len(unique_labels))]
    
    #fig.ax_heatmap.legend(handles=h, labels=list(unique_labels), title="Conditions")

    #for hidden in h:
    #    hidden.set_visible(False)
    if title is not None:
        fig.fig.suptitle(title)
    #fig.fig.tight_layout()
    plt.show()
    
    if save_path is not None:
        fig.figure.savefig(save_path)

    return linkage


def hier_clustering(dist,labels, ax=None, cluster=True, method="average",title=None,dist_name="",log=False,save_path=None,cmap="tab20",hue_order=None,annotation=False):
    #Creating list of colors for conds
    unique_inds = np.unique(labels, return_index=True)[1]
    unique_labels = np.asarray([labels[index] for index in sorted(unique_inds)] ).tolist() if hue_order is None else hue_order
    if isinstance(cmap,str):
        cmap = sns.color_palette(palette=cmap,n_colors=len(unique_labels))
        colors = [cmap[unique_labels.index(l)] for l in labels]
    else:
        colors = [cmap[l] for l in labels]

    dist = np.copy(dist)

    if cluster:
        dist[np.eye(len(dist),dtype=bool)] = 0  
        linkage =  hc.linkage(sp.distance.squareform(dist), method=method, optimal_ordering=True)
    else:
        linkage = None

    norm = None
    if log:
        norm = LogNorm()
        dist[dist<=0] = np.min(dist[dist>0])

    
    fig = sns.clustermap(dist, figsize=(5,5),row_cluster=cluster, col_cluster=cluster, row_linkage=linkage ,col_linkage=linkage, dendrogram_ratio=0.15,
            row_colors=colors, col_colors=colors,method=method, cmap=sns.cm.rocket_r,
            #cbar_pos=(.2, 0, .2, .02),cbar_kws={"label":dist_name, "orientation":'horizontal'},yticklabels=False,xticklabels=False) #,)
            cbar_pos=(0.05, 0.1, .1, .02),cbar_kws={"orientation":'horizontal'},yticklabels=False,xticklabels=annotation,norm=norm)

    fig.ax_heatmap.tick_params(right=False, bottom=False if annotation is False else True) #sounds really stupid but annotation might be non-boolean


    fig.ax_col_dendrogram.set_visible(False)

    fig.ax_cbar.set_title(dist_name)

    if title is not None:
        fig.fig.suptitle(title)

    plt.show()
    
    if save_path is not None:
        fig.figure.savefig(save_path)

    return linkage