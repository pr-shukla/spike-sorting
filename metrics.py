import numpy as np
from numpy.core.numeric import identity
from scipy.spatial.distance import mahalanobis

def accuracy_metrics(prediction_label, 
                     groundtruth_label):

    '''
    '''

    final_output = {}

    different_prediction_labels = []
    different_groundtruth_labels = []

    
    different_prediction_labels = list(range(min(prediction_label),max(prediction_label)+1))
    different_groundtruth_labels = list(range(min(groundtruth_label),max(groundtruth_label)+1))
    #print(different_groundtruth_labels,different_prediction_labels)
    for i in different_groundtruth_labels:

        groundtruth_accuracy_dict = {}
        groundtruth_precision_dict = {}
        groundtruth_recall_dict = {}
        prediction_corresponding_i_unit = np.array(prediction_label)[groundtruth_label==i]

        for j in different_prediction_labels:

            
            n1 = (len(prediction_corresponding_i_unit)-(list(prediction_corresponding_i_unit).count(j)))
            n2 = list(prediction_corresponding_i_unit).count(j)
            n3 = ((list(prediction_label).count(j))-n2)

            try:
                acc = n2/(n1+n2+n3)
            except:
                acc = 0
            
            try:
                prec = n2/(n2+n3)
            except:
                prec = 0
            
            try:
                recall = n2/(n1+n2)
            except:
                recall = 0

            groundtruth_accuracy_dict[j] = acc
            groundtruth_precision_dict[j] = prec
            groundtruth_recall_dict[j] = recall

        #print(groundtruth_accuracy_dict)
        max_acc = max(groundtruth_accuracy_dict.values())
        max_acc_idx = list(groundtruth_accuracy_dict.values()).index(max_acc)

        max_acc_prec = list(groundtruth_precision_dict.values())[max_acc_idx]
        max_acc_recall = list(groundtruth_recall_dict.values())[max_acc_idx]

        num_spikes = list(groundtruth_label).count(i)
        final_output['Unit '+str(i)] = {'acc':max_acc,
                                        'prec':max_acc_prec,
                                        'recall':max_acc_recall}#,'spike_rate':num_spikes*10**7/max_time}

    return final_output


def isolation_distance(X, labels):

    '''
    '''

    max_label = max(labels)

    output = {}

    for i in range(1,max_label+1):

        num_spikes_with_label_i = list(labels).count(i)

        #print(i,num_spikes_with_label_i)

        if num_spikes_with_label_i == 0:
            continue

        X_with_label_i = X[labels==i]

        mean_X_with_label_i = np.sum(X_with_label_i,axis=0)/num_spikes_with_label_i

        X_not_with_label_i = X[labels!=i]

        dist_X_not_with_label_i_mean = list(np.sum(np.square(X_not_with_label_i-mean_X_with_label_i),axis=1))

        shape_X_with_label_i = np.shape(X_with_label_i)

        nearest_X = np.zeros(shape_X_with_label_i)

        for j in range(num_spikes_with_label_i):

            min_distance = min(dist_X_not_with_label_i_mean)
            idx_min_distance = dist_X_not_with_label_i_mean.index(min_distance)

            nearest_X[j] = X_not_with_label_i[idx_min_distance]

            dist_X_not_with_label_i_mean[idx_min_distance] = max(dist_X_not_with_label_i_mean)

        new_cluster = np.concatenate([X_with_label_i,nearest_X])

        #mean_new_cluster = np.sum(new_cluster,axis=0)/(2*num_spikes_with_label_i)
        mean_new_cluster = np.sum(X_with_label_i,axis=0)/(num_spikes_with_label_i)

        #print(np.cov(new_cluster))#+10**-3)
        mahalnobis_dist = 0
        
        #identity_matrix = np.eye(2*num_spikes_with_label_i,2*num_spikes_with_label_i)
        #inv_cov_cluster = np.linalg.inv(np.cov(new_cluster.T))#,identity_matrix)[0]#+10**1)
        
        try:
            inv_cov_cluster = np.linalg.inv(np.cov(X_with_label_i.T))#,identity_matrix)[0]#+10**1)
        except:
            mahalnobis_dist = 0
            continue
        #print(inv_cov_cluster)
        #print('Unit ', str(i))
        #print(nearest_X[num_spikes_with_label_i-1],mean_new_cluster)
        #mahalnobis_dist = mahalanobis(nearest_X[num_spikes_with_label_i-1],mean_new_cluster,inv_cov_cluster)
        #print(mahalnobis_dist)
        try:
            mahalnobis_dist = mahalanobis(nearest_X[num_spikes_with_label_i-1],mean_new_cluster,inv_cov_cluster)

        except:
            mahalnobis_dist = 0
        
        #print(mahalnobis_dist)
        output['Unit '+str(i)] = {'iso_dist':mahalnobis_dist}

    return output

if __name__ == '__main__':

    x = np.array([[1,1],[1.5,1.5],[1.5,1],[100,100],[104.5,104.5],[104.5,104],[105,105]])
    labels = np.array([1,1,1,2,2,2,2])

    out = isolation_distance(x,labels)

    print(out)
