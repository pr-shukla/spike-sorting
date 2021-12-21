import numpy as np


def isosplit(X, opts):

    '''
    '''

    T_find_best_pair=0
    T_find_centroids=0
    T_attempt_redistribution=0
    T_isosplit1d=0
    T_projection=0
    T_sort=0

    opts.isocut_threshold=0.9
    opts.min_cluster_size=10
    opts.K=25
    opts.minsize=3
    opts.max_iterations_per_number_clusters=5000
    opts.verbose=0
    opts.return_iterations=0

    (M,N)=np.shape(X)

    labels = local_kmeans_sorber(X,opts.K)

    centroids=compute_centroids(X,labels)
    distances=compute_distances(centroids)

    num_iterations_with_same_number_of_clusters=0

    attempted_redistributions=list(np.zeros(0,1))

    num_iterations=0

    while True:

        num_iterations=num_iterations+1
        old_labels = labels

        (label1, label2) = find_best_pair(X,labels,centroids,distances,attempted_redistributions)

        if (label1==0):
            break

        inds1=np.where(labels==label1)
        inds2=np.where(labels==label2)

        centroid1=centroids[label1,:]
        centroid2=centroids[label2,:]

        (ii1,ii2,redistributed,inf0)=attempt_to_redistribute_two_clusters(X,
                                                                          inds1,
                                                                          inds2,
                                                                          centroid1,
                                                                          centroid2,
                                                                          opts.split_threshold,
                                                                          opts)

        T_projection=T_projection+inf0.T_projection
        T_isosplit1d=T_isosplit1d+inf0.T_isosplit1
        T_sort=T_sort+inf0.T_sort

        if (len(ii2)>0):
            num_iterations_with_same_number_of_clusters=num_iterations_with_same_number_of_clusters+1
        else:
            num_iterations_with_same_number_of_clusters=0
        
        if (distances(label1,label2)==np.inf):

            break

        attempted_redistributions.append(distances(label1,label2))

        if redistributed:

            labels[ii1]=label1
            labels[ii2]=label2

            centroids[label1,:]=np.mean(X[ii1,:],2)
            





    if len(opts.initial_labels)==0:
        target_parcel_size=opts.min_cluster_size
        target_num_parcels=opts.K_init
        data.labels=parcelate2(X,target_parcel_size,target_num_parcels,struct('final_reassign',0))
        Kmax=max(data.labels)


def distances(label1,label2):
    return

def local_kmeans_sorber(X,k):
    return 

def compute_centroids(X, labels):
    return

def compute_distances(centroids):
    return

def find_best_pair(X,labels,centroids,distances,attempted_redistributions):
    return

def attempt_to_redistribute_two_clusters(X,inds1,inds2,centroid1,centroid2,opts.split_threshold,opts):
    return