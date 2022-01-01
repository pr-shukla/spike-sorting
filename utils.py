import json

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift

from metrics import isolation_distance, accuracy_metrics

# Number of Channels in tetrode

num_channels = 4

# Number of samples for single spike

num_samples = 64

# Sampling frequency

freq_sampling = 30000

def read_mat_file(filename):
    
    '''
    Read .mat file

    Parameters
    ----------
    filename: str
        Name of the .mat file to be read

    Returns
    -------
    mat: dict
        Dictionary with keys as parameters and values
        are parameter value
    '''

    filepath = filepath = 'F:/Neuroscience/data/' + filename + '.mat'

    # Reading .mat file

    mat = scipy.io.loadmat(filepath)

    return mat

def read_binary_file(filename):

    '''
    '''

    filepath = 'F:/Neuroscience/data/' + filename

    file = open(filepath, 'rb')


    return file

def spike_times(duration_in_hrs, 
                filename):

    '''
    Read SpikeTimes file

    Parameters
    ----------
    duration_in_hrs: float
        Time duration upto which spikes need to be
        considered
    filename: str
        Name of the file to be read

    Returns
    -------
    time_series: numpy.array
        Time Bin at which spikes occured upto time
        duration_in_hrs
    '''

    # convert time from hrs to seconds

    duration_in_sec = duration_in_hrs*3600

    # Number of samples corresponding to 30 kHz frequency

    num_samples = 30000*duration_in_sec

    filepath = 'F:/Neuroscience/data/' + filename

    # Reading binary file with data stored in uint64 dtype

    spike_time_data = np.fromfile(filepath, dtype=np.uint64)

    # Count number of samples 'i' upto given time duration

    for i in range(len(spike_time_data)):
        if spike_time_data[i] >num_samples:
            break
    
    # Returns samples upto ith bin

    return spike_time_data[:i]

def spike_data_from_channels(no_of_spikes, 
                             filename):

    '''
    Reads binary file for simulation spike data and
    convert data N*256 matrix where each row will be
    spike samples recorded in each channel

    Parameters
    ----------
    no_of_spikes: int
        Number of spikes occured in given time
        duration
    filename: str
        Name of spike data file to be read

    Returns
    -------

    '''

    no_of_samples = no_of_spikes*num_channels*num_samples

    filepath = 'F:/Neuroscience/data/' + filename

    data = np.fromfile(filepath, dtype=np.int16, count=no_of_samples)

    channel1_data = []
    channel2_data = []
    channel3_data = []
    channel4_data = []
    spike_data_4_channel = []

    # Store spike data for each channel in seperate
    # list

    for i in range(0,no_of_samples,4):

        channel1_data.append(data[i])
        channel2_data.append(data[i+1])
        channel3_data.append(data[i+2])
        channel4_data.append(data[i+3])

    # Store spike data in  N*256 dim matrix
    # where each row corresponds to different
    # spikes and columns corresponds to data
    # recorded on 4 channels seprately by
    # dividing data in the 64 equal chunks
     
    for i in range(no_of_spikes):

        shift_idx = 0

        # Start and end bin index for given spikes

        start_idx = int(num_samples*(i+shift_idx))
        end_idx = int(num_samples*(i+1+shift_idx))

        try:

            #print(len(channel1_data[start_idx:(end_idx)]))
            if len(channel1_data[start_idx:(end_idx)]) == 64:
                spike_data_4_channel.append(channel1_data[start_idx:(end_idx)]+
                                            channel2_data[start_idx:(end_idx)]+
                                            channel3_data[start_idx:(end_idx)]+
                                            channel4_data[start_idx:(end_idx)])
        
        except:
            continue

    # Convert all the spike data from list to array, it will facilitate
    # lots of numpy array functionalities that will be later helpful  

    channel1_data  =np.array(channel1_data)
    channel2_data  =np.array(channel2_data)
    channel3_data  =np.array(channel3_data)
    channel4_data  =np.array(channel4_data)

    spike_data_4_channel = np.array(spike_data_4_channel)

    return channel1_data, channel2_data, channel3_data, channel4_data, spike_data_4_channel

def avg_spikes_channels(ch1, ch2, ch3, ch4):

    '''
    '''

    return (ch1+ch2+ch3+ch4)/4

def avg_multiple_spikes(spike_data, no_of_spikes):

    '''
    '''

    avg_spikes = np.zeros(num_samples)

    for i in range(no_of_spikes):

        #print(num_samples*(i+1/2), num_samples*(i+3/2))
        start_idx = int(num_samples*(i))#+1/2))
        end_idx = int(num_samples*(i+1))#3/2))

        try:
            avg_spikes = (avg_spikes+spike_data[start_idx:end_idx])
        except:
            continue

    avg_spikes = avg_spikes/no_of_spikes

    return avg_spikes

def gmm_feature(features_array, 
                num_peaks):

    '''
    '''

    gmm = GaussianMixture(n_components=num_peaks, 
                          random_state=0).fit(features_array)

    return gmm

def peaks_gmm_model(features_array,
                    gmm_object):

    '''
    '''

    feature_max_value = max(features_array)
    feature_min_value = min(features_array)

    feature_array = np.linspace(feature_min_value, feature_max_value, 100)
    feature_probs = gmm_object.predict_proba(feature_array)

    plt.plot(feature_probs)
    plt.show()

    return feature_probs

def jsonify_simulation_labels(num_spikes,
                              spike_time_data,
                              ground_truth_time,
                              ground_truth):

    '''
    '''

    # this index used to reduce the search for
    # simulation spike time, once the spike time
    # has been searched it is not possible that 
    # next spike will occur at previous time bins
    # because spikes will occur in increasing
    # time order only.
     
    idx_readed = 0

    # This list contains final labels for spike
    # i.e. labels for simulation spike will be
    # either ground truth spike label if it occurs
    # in certain range of original spike time
    # or it will be labeled as zero i.e. noise
     
    ground_noise_label = []

    # Iterating over number of spike events in given
    # duration of time

    for i in range(num_spikes):

        # Simulation Time of current spike 

        simulation_time = spike_time_data[i]

        # Default label for current spike as true
        # if it does not gets any other label

        spike_is_noise = True
        
        # Counting number of iterations completed
        # for searching spike in ground_truth spike
        # time

        j = 0

        # Iterating over ground truth spike times
         
        for real_time in ground_truth_time[idx_readed:]:

            # Count increases by 1 as one iteration for seaching spike 
            # started

            j += 1

            # If difference of samples between simulation and real time is 
            # less then 2 samples, then spikes will be considered as true
            # spike 

            if np.abs(simulation_time-real_time) < 2:

                # As simulatio spike has been detected as true spike
                # it should be given label from the ground truth 
                
                ground_noise_label.append(int(ground_truth[ground_truth_time.index(real_time)]))
                
                # Update the index upto which spike time has already been
                # checked
                
                idx_readed = ground_truth_time.index(real_time)
                
                # As spike has been detected as true spike it will get noise label
                # as false and inner for loop will break

                spike_is_noise = False
                break

            # If real time has exceeded above simulation time
            # then there is no more possibility of finding
            # ground truth corresponding to simulation spike
            # and the loop will break

            elif real_time>simulation_time:
                break
        
        print('Number of iterations completed:', j, i)
        
        # Give spike label zero if it is noise

        if spike_is_noise:
            #print('Hitted Noise')
            ground_noise_label.append(0)

    #print('Ground Noise Level:',ground_noise_label)
    #print('DataSet Ground Labels without Noise:', ground_truth[:num_spikes])

    # Store labels of the spikes in dictionary

    ground_noise_label_dict = {'Ground_Noise_Label': ground_noise_label}
    #print(ground_noise_label_dict)

    # create json object from the spike labels
    
    json_object = json.dumps(ground_noise_label_dict)
  
    # Writing to sample.json

    # Write json file from json object

    with open("ground_noise_label.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':

    # Time duration to be considered

    time_duration_hrs = 1
    
    # Whether to jsonify label or not

    jsonify_output = False

    # Time duration for 1 processing chunk

    time_duration_chunk = 0.01

    # Number of samples in total time duration

    num_samples_total_time = time_duration_hrs*3600*30000

    # Number of samples for 1 chunk 

    num_samples_one_chunk = time_duration_chunk*3600*30000

    # Number of iterations to comlete total time bu chunks

    if num_samples_total_time%num_samples_one_chunk == 0:

        num_iterations = int(num_samples_total_time//num_samples_one_chunk)

    else:

        num_iterations = int(num_samples_total_time//num_samples_one_chunk + 1)
    
    print('Number of iterations required for complete spike sorting:', num_iterations)
    # Call function to read spike time data

    spike_time_data = spike_times(time_duration_hrs,
                                  'SpikeTimes')
    
    # Call function to read parameters file

    mat_data = read_mat_file('dataset_params')

    print('Parameters of dataset:', mat_data.keys())
    
    # Ground truth labels for true spikes

    ground_truth = mat_data['sp_u'][0][0][0]#[:num_spikes]
    
    # Ground truth time for true spikes

    ground_truth_time = list(mat_data['sp'][0][0][0])

    # Number of spikes corresponding to given time duration

    num_spikes = len(spike_time_data)

    print('Number of spikes in',time_duration_hrs, 'hours duration are', num_spikes)

    # Jsonify ground truth labels for simulation spikes

    if jsonify_output:
        jsonify_simulation_labels(num_spikes,
                                  spike_time_data,
                                  ground_truth_time,
                                  ground_truth)

    # Read spike waveforms from binary file 
    # and convert waveform data into N*256
    # matrix

    ch1_data, ch2_data, ch3_data, ch4_data,spike_data_4_channel = spike_data_from_channels(num_spikes,
                                                                                           'Spikes')

    # Scale down spike waveforms by 10

    spike_data_4_channel = spike_data_4_channel/10

    # Averaged Accuracy over all time

    avg_acc_all_time = 0

    # Read data corresponding to labels of simulation spikes
     
    file = open('ground_noise_label.json')
    ground_noise_label = json.load(file)['Ground_Noise_Label']

    # convert labels from list to array

    ground_noise_label = np.array(ground_noise_label)
    
    # Number of Spikes sorted

    num_spikes_sorted = 0

    # Iterating to do spike sorting on chunks of data

    for i in range(num_iterations):

        # Spike Time Data for a chunk in iteration

        spike_time_data_single_chunk = spike_time_data[np.all([i*num_samples_one_chunk<=spike_time_data,
                                                               spike_time_data<(i+1)*num_samples_one_chunk],axis=0)]

        # Number of spikes in current chunk

        num_spikes_single_chunk = len(spike_time_data_single_chunk)

        print('Number of Spikes in given chunk:', num_spikes_single_chunk)

        # Spike Data for single chunk

        spike_data_4_channel_single_chunk = spike_data_4_channel[num_spikes_sorted:
                                                                 num_spikes_sorted+num_spikes_single_chunk]

        # Ground Labels of Spikes for current chunk

        ground_noise_label_single_channel = ground_noise_label[num_spikes_sorted:
                                                               num_spikes_sorted+num_spikes_single_chunk]

        # Number of Spikes sorted

        num_spikes_sorted += num_spikes_single_chunk

        # create object for 8 PCA components

        pca = PCA(n_components = 8)

        # Create PCA model with spike data

        pca.fit(spike_data_4_channel_single_chunk)

        # Convert 256 dim spike data into 8 dim
        # PCA components
        
        Y = pca.fit_transform(spike_data_4_channel_single_chunk)

        # Filter data which is not noise and others labels need
        # to be filtered as per requirements 

        Y_ground_truth = Y[np.all([ ground_noise_label_single_channel!=0,
                                    ground_noise_label_single_channel!=8,
                                    ground_noise_label_single_channel!=7], axis = 0)]
        Y_ground_label = ground_noise_label_single_channel[np.all([
                                            ground_noise_label_single_channel!=0,
                                            ground_noise_label_single_channel!=8,
                                            ground_noise_label_single_channel!=7], axis = 0)]

        # Filter noise only, any one of the data will be used from above one
        # and the this one

        Y_ground_truth = Y[ground_noise_label_single_channel!=0]
        Y_ground_label = ground_noise_label_single_channel[ground_noise_label_single_channel!=0]

        # Calculate isolation distance corresponding to ground truth cluster
        # labels

        groundtruth_iso_dist = isolation_distance(Y_ground_truth,Y_ground_label)
        #print(groundtruth_iso_dist)
        #gm = GaussianMixture(n_components=8 , random_state=0).fit(Y_ground_truth)

        #labels = gm.predict(Y_ground_truth)

        # Create model for clustering
        clustering = MeanShift(bandwidth=100).fit(Y_ground_truth)
        
        # Calculate prediction labels using clustering model

        labels = clustering.labels_

        #print('Shape of spike data',np.shape(Y))

        # Calculate accuracy of the prediction using ground
        # truth labels and prediction labels
    
        acc_metrics = accuracy_metrics(labels, Y_ground_label)
        #print('Accuracy of the model:',acc_metrics)

        # Calculate isolation distance for prediction label clusters

        prediction_iso_dist = isolation_distance(Y_ground_truth,labels)
        #print('Isolation Distance:',prediction_iso_dist)
        
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
        

    #print(len(ch1_data), len(ch2_data), len(ch3_data), len(ch4_data))

    #print('Shape of spike data',np.shape(spike_data_4_channel))

    #avg_ch_data = avg_spikes_channels(ch1_data,
    #                                ch2_data,
    #                                ch3_data,
    #                                ch4_data)

    #avg_multi_spikes = avg_multiple_spikes(avg_ch_data,
    #                                    num_spikes)

    # create object for 8 PCA components
    '''
    pca = PCA(n_components = 8)

    # Create PCA model with spike data

    pca.fit(spike_data_4_channel)

    # Convert 256 dim spike data into 8 dim
    # PCA components
    
    Y = pca.fit_transform(spike_data_4_channel)
'''
    #num_peaks_feature_gmm = 8
    #feature_array = np.array([Y[:,0]])
    #feature_array = np.reshape(feature_array,(len(Y[:,0]),1))
    #feature_gmm_object = gmm_feature(feature_array, 
    #                                 num_peaks_feature_gmm)
    
    #peaks_feature_gmm_model = peaks_gmm_model(feature_array,
    #                                          feature_gmm_object)

    #gm = GaussianMixture(n_components=7 , random_state=0).fit(Y)

    #labels = gm.predict(Y)

    #X_axis = np.array([np.linspace(-3000,6000,100)])
    #Y_axis = np.array([np.linspace(-3000,5000)]).T

    #Z_axis = 

    
    '''
    
    for i in range(num_spikes):

        simulation_time = spike_time_data[i]

        spike_is_noise = True
        j = 0

        for real_time in ground_truth_time[idx_readed:]:

            j += 1

            #print(simulation_time, real_time, np.abs(simulation_time-real_time))
            if np.abs(simulation_time-real_time) < 2:

                ground_noise_label.append(int(ground_truth[ground_truth_time.index(real_time)]))
                idx_readed = ground_truth_time.index(real_time)
                spike_is_noise = False
                break
            elif real_time>simulation_time:
                break
        
        print('Number of iterations completed:', j, i)
        if spike_is_noise:
            #print('Hitted Noise')
            ground_noise_label.append(0)

    #print('Ground Noise Level:',ground_noise_label)
    #print('DataSet Ground Labels without Noise:', ground_truth[:num_spikes])

    ground_noise_label_dict = {'Ground_Noise_Label': ground_noise_label}
    #print(ground_noise_label_dict)

    
    json_object = json.dumps(ground_noise_label_dict)
  
    # Writing to sample.json
    with open("ground_noise_label.json", "w") as outfile:
        outfile.write(json_object)
'''
    
    '''
    
    # Filter data which is not noise and others labels need
    # to be filtered as per requirements 

    Y_ground_truth = Y[np.all([ ground_noise_label!=0,
                                ground_noise_label!=8,
                                ground_noise_label!=7], axis = 0)]
    Y_ground_label = ground_noise_label[np.all([
                                        ground_noise_label!=0,
                                        ground_noise_label!=8,
                                        ground_noise_label!=7], axis = 0)]

    # Filter noise only, any one of the data will be used from above one
    # and the this one

    Y_ground_truth = Y[ground_noise_label!=0]
    Y_ground_label = ground_noise_label[ground_noise_label!=0]

    # Calculate isolation distance corresponding to ground truth cluster
    # labels

    groundtruth_iso_dist = isolation_distance(Y_ground_truth,Y_ground_label)
    print(groundtruth_iso_dist)
    #gm = GaussianMixture(n_components=8 , random_state=0).fit(Y_ground_truth)

    #labels = gm.predict(Y_ground_truth)

    # Create model for clustering
    clustering = MeanShift(bandwidth=100).fit(Y_ground_truth)
    
    # Calculate prediction labels using clustering model

    labels = clustering.labels_

    print('Shape of spike data',np.shape(Y))

    # Calculate accuracy of the prediction using ground
    # truth labels and prediction labels
 
    acc_metrics = accuracy_metrics(labels, Y_ground_label,max(ground_truth_time[:num_spikes]))
    print('Accuracy of the model:',acc_metrics)

    # Calculate isolation distance for prediction label clusters

    prediction_iso_dist = isolation_distance(Y_ground_truth,labels)
    print('Isolation Distance:',prediction_iso_dist)
    
    # Calculate avg accuracy of all the clusters

    avg_acc = 0
    for unit_acc in acc_metrics:

        avg_acc += acc_metrics[unit_acc]['acc']

    avg_acc /= len(acc_metrics)

    print('Avg Accuracy:', avg_acc)

    # Create list for accuracy metrics and isolation distance 
    # metric to make scatter plot of metrics

    acc_list = []
    spike_rate_list = []
    iso_dist_list = []
    for unit_acc in acc_metrics:

        print(unit_acc,':',acc_metrics[unit_acc])

        acc_list.append(acc_metrics[unit_acc]['acc'])
        #spike_rate_list.append(acc_metrics[unit_acc]['spike_rate'])
        iso_dist_list.append(np.log10(groundtruth_iso_dist[unit_acc]['iso_dist']))
'''

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