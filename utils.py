import json

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from feature import FeatureSelection

from metrics import Metrics
from cluster import Cluster
from spikes import SpikeTime, SpikeLabel, SpikeWave

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

    return spike_data_4_channel

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

    print('Utils python script executed')