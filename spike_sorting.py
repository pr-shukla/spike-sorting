import json
import warnings

import numpy as np
import matplotlib.pyplot as plt

from feature import FeatureSelection
from metrics import Metrics
from cluster import Cluster
from spikes import SpikeTime, SpikeLabel, SpikeWave
from utils import spike_times,read_mat_file,spike_data_from_channels

def main(time,
         processing_time_chunk,
         waveform_feature_scaling=1,
         pca_components=None,
         meanshift_band_width=None
         ):
    
    '''
    Main code executing spike sorting algorithm

    Parameters
    ----------
    time: np.array

    processing_time_chunk: np.array

    waveform_feature_scaling: int

    pca_components: int

    meanshift_band_width: int
    '''
                  
    spike_time_object = SpikeTime(time_duration=time,
                                  processing_time_duration=processing_time_chunk,
                                  sampling_frequency=30000)

    # Call function to read spike time data

    spike_time_object.spikes_occurence_time = spike_times(spike_time_object.time_duration,
                                                          'SpikeTimes')
    
    # Call function to read parameters file

    mat_data = read_mat_file('dataset_params')

    spike_time_object.spikes_true_time = list(mat_data['sp'][0][0][0])

     # Read data corresponding to labels of simulation spikes
     
    file = open('ground_noise_label.json')
    ground_noise_label = json.load(file)['Ground_Noise_Label']

    spike_label_object = SpikeLabel(labels_without_noise=mat_data['sp_u'][0][0][0],
                                    labels_with_noise=ground_noise_label)

    
    # Whether to jsonify label or not

    jsonify_output = False

    # Number of iterations to comlete total time bu chunks

    if int(spike_time_object.num_samples%spike_time_object.num_samples_processing) == 0:

        num_iterations = int(spike_time_object.num_samples//spike_time_object.num_samples_processing)

    else:

        num_iterations = int(spike_time_object.num_samples//spike_time_object.num_samples_processing + 1)
    
    print('Number of iterations required for complete spike sorting:', num_iterations)
    
    #print('Parameters of dataset:', mat_data.keys())

    # Number of spikes corresponding to given time duration

    num_spikes = spike_time_object.no_of_spikes()

    print('Number of spikes in',spike_time_object.time_duration, 'hours duration are', num_spikes)

    # Jsonify ground truth labels for simulation spikes

    if jsonify_output:

        spike_label_object.jsonify_true_noise_labels(num_spikes=num_spikes,
                                                     spikes_detection_time=spike_time_object.spikes_occurence_time,
                                                     spikes_true_time=spike_time_object.spikes_true_time)
        '''jsonify_simulation_labels(num_spikes,
                                  spike_time_data,
                                  ground_truth_time,
                                  ground_truth)
'''
    # Read spike waveforms from binary file and convert waveform data into N*256 matrix

    spike_data_4_channel = spike_data_from_channels(spike_time_object.num_spikes,'Spikes')

    # Create Spike Waveform object

    spike_waveform_object = SpikeWave(spike_data_from_channels(spike_time_object.num_spikes,'Spikes'),
                                      feature_scaling=waveform_feature_scaling)

    # Scale down spike waveforms by 10

    spike_data_4_channel = spike_data_4_channel/10

    # Averaged Accuracy over all time

    avg_acc_all_time = 0

    # Read data corresponding to labels of simulation spikes
     
    file = open('ground_noise_label.json')
    ground_noise_label = json.load(file)['Ground_Noise_Label']

    # convert labels from list to array

    ground_noise_label = np.array(ground_noise_label)

    spike_label_object.true_noise_spikes_labels = np.array(spike_label_object.true_noise_spikes_labels)
    
    # Number of Spikes sorted

    num_spikes_sorted = 0

    # Iterating to do spike sorting on chunks of data

    for i in range(num_iterations):

        # Spike Time Data for a chunk in iteration

        
        spike_time_data_single_chunk = spike_time_object.spikes_occurence_time[
                    np.all([i*spike_time_object.num_samples_processing<=spike_time_object.spikes_occurence_time,
                            spike_time_object.spikes_occurence_time<(i+1)*spike_time_object.num_samples_processing],
                            axis=0)]

        # Number of spikes in current chunk

        num_spikes_single_chunk = len(spike_time_data_single_chunk)

        print('Number of Spikes in given chunk:', num_spikes_single_chunk)

        # Spike Data for single chunk

        spike_data_4_channel_single_chunk = spike_waveform_object.spikes_waveform[num_spikes_sorted:
                                                        num_spikes_sorted+num_spikes_single_chunk]

        # Ground Labels of Spikes for current chunk

        ground_noise_label_single_channel = spike_label_object.true_noise_spikes_labels[num_spikes_sorted:
                                                        num_spikes_sorted+num_spikes_single_chunk]

        # Number of Spikes sorted

        num_spikes_sorted += num_spikes_single_chunk

        # create object for PCA components

        featurization_object = FeatureSelection()

        featurization_object.pca_model(spike_data_4_channel_single_chunk,
                                       num_pca_components=pca_components)

        # Convert 256 dim spike data into 8 dim PCA components
        
        Y = featurization_object.pca_features(spike_data_4_channel_single_chunk)

        # Filter data which is not noise and others labels need to be filtered as per requirements 

        Y_ground_truth = Y[np.all([ ground_noise_label_single_channel!=0,
                                    ground_noise_label_single_channel!=8,
                                    ground_noise_label_single_channel!=7], axis = 0)]
        Y_ground_label = ground_noise_label_single_channel[np.all([
                                            ground_noise_label_single_channel!=0,
                                            ground_noise_label_single_channel!=8,
                                            ground_noise_label_single_channel!=7], axis = 0)]

        # Filter noise only, any one of the data will be used from above one and the this one

        Y_ground_truth = Y[ground_noise_label_single_channel!=0]
        Y_ground_label = ground_noise_label_single_channel[ground_noise_label_single_channel!=0]

        # Calculate isolation distance corresponding to ground truth cluster labels

        metrics_object = Metrics()

        groundtruth_iso_dist = metrics_object.isolation_distance(Y_ground_truth,Y_ground_label)
        
        # Create model for clustering and Calculate prediction labels using clustering model

        cluster_object = Cluster()

        cluster_model,labels = cluster_object.meanshift_model(X=Y_ground_truth,
                                                              estimated_cluster_size=meanshift_band_width)
        
        # Calculate accuracy of the prediction using ground truth labels and prediction labels
    
        acc_metrics = metrics_object.accuracy_metrics(labels, Y_ground_label)
        #print('Accuracy of the model:',acc_metrics)

        # Calculate isolation distance for prediction label clusters

        prediction_iso_dist = metrics_object.isolation_distance(Y_ground_truth,labels)

        # Calculate avg accuracy of all the clusters

        avg_acc = 0
        for unit_acc in acc_metrics:

            avg_acc += acc_metrics[unit_acc]['acc']

        avg_acc /= len(acc_metrics)

        print('Avg Accuracy:', avg_acc)

        avg_acc_all_time += avg_acc

        # Create list for accuracy metrics and isolation distance 
        # metric to make scatter plot of metrics
        '''
        acc_list = []
        spike_rate_list = []
        iso_dist_list = []
        for unit_acc in acc_metrics:

            print(unit_acc,':',acc_metrics[unit_acc])

            acc_list.append(acc_metrics[unit_acc]['acc'])
            #spike_rate_list.append(acc_metrics[unit_acc]['spike_rate'])
            iso_dist_list.append(np.log10(groundtruth_iso_dist[unit_acc]['iso_dist']))
'''
    avg_acc_all_time /= num_iterations

    print('Average Accuracy for given overall time:', avg_acc_all_time)
        
    #avg_ch_data = avg_spikes_channels(ch1_data,
    #                                ch2_data,
    #                                ch3_data,
    #                                ch4_data)

    #avg_multi_spikes = avg_multiple_spikes(avg_ch_data,
    #                                   


    plt.subplot(2,1,1)
    plt.scatter(Y_ground_truth[:,0], Y_ground_truth[:,1], c = labels)#facecolors='none', edgecolors=labels)
    plt.title('Prediction')
    plt.subplot(2,1,2)
    plt.scatter(Y_ground_truth[:,0], Y_ground_truth[:,1], c = Y_ground_label)#facecolors='none', edgecolors=ground_noise_label)#c = ground_noise_label )
    plt.title('Ground Truth')
    plt.show()

    #plt.scatter(iso_dist_list, acc_list)
    #plt.xlabel('Isolation Distance in log scale')
    #plt.ylabel('Accuracy')
    #plt.show()

    #plt.plot(ch3_data[:64])
    #plt.show()

if __name__ == '__main__':

    

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(time=1,
            processing_time_chunk=0.01,
            waveform_feature_scaling=10,
            pca_components=8,
            meanshift_band_width=100)
