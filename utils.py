import json

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift

from metrics import isolation_distance, accuracy_metrics

num_channels = 4
num_samples = 64

def read_mat_file(filename):
    '''
    '''

    filepath = filepath = 'F:/Neuroscience/data/' + filename + '.mat'

    mat = scipy.io.loadmat(filepath)

    return mat



def read_binary_file(filename):

    '''
    '''

    filepath = 'F:/Neuroscience/data/' + filename

    file = open(filepath, 'rb')


    return file

def spike_times(duration_in_hrs, filename):

    '''
    '''

    duration_in_sec = duration_in_hrs*3600
    num_samples = 30000*duration_in_sec

    filepath = 'F:/Neuroscience/data/' + filename

    spike_time_data = np.fromfile(filepath, dtype=np.uint64)

    for i in range(len(spike_time_data)):
        if spike_time_data[i] >num_samples:
            break
    
    return spike_time_data[:i]

def spike_data_from_channels(no_of_spikes, filename):

    '''
    '''

    no_of_samples = no_of_spikes*num_channels*num_samples

    filepath = 'F:/Neuroscience/data/' + filename

    data = np.fromfile(filepath, dtype=np.int16, count=no_of_samples)

    channel1_data = []
    channel2_data = []
    channel3_data = []
    channel4_data = []
    spike_data_4_channel = []

    for i in range(0,no_of_samples,4):

        channel1_data.append(data[i])
        channel2_data.append(data[i+1])
        channel3_data.append(data[i+2])
        channel4_data.append(data[i+3])

    for i in range(no_of_spikes):

        shift_idx = 0

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

    channel1_data  =np.array(channel1_data)
    channel2_data  =np.array(channel2_data)
    channel3_data  =np.array(channel3_data)
    channel4_data  =np.array(channel4_data)

    spike_data_4_channel = np.array(spike_data_4_channel)
    #print(spike_data_4_channel)

    return channel1_data, channel2_data, channel3_data, channel4_data, spike_data_4_channel

def jsonify_output():

    '''
    '''

    return
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


if __name__ == '__main__':

    time_duration_hrs = 10
    spike_time_data = spike_times(time_duration_hrs,
                                  'SpikeTimes')

    num_spikes = len(spike_time_data)

    print('Number of spikes in',time_duration_hrs, 'hours duration are', num_spikes)

    ch1_data, ch2_data, ch3_data, ch4_data,spike_data_4_channel = spike_data_from_channels(num_spikes,
                                                                                           'Spikes')


    spike_data_4_channel = spike_data_4_channel/10

    mat_data = read_mat_file('dataset_params')

    print('Parameters of dataset:', mat_data.keys())
    print('Parameter Units lenght', mat_data['sp_u'][0][0][0])
    
    ground_truth = mat_data['sp_u'][0][0][0]#[:num_spikes]
    ground_truth_time = list(mat_data['sp'][0][0][0])

    ground_noise_label = []
    idx_readed = 0

    #print(len(ch1_data), len(ch2_data), len(ch3_data), len(ch4_data))

    #print('Shape of spike data',np.shape(spike_data_4_channel))

    #avg_ch_data = avg_spikes_channels(ch1_data,
    #                                ch2_data,
    #                                ch3_data,
    #                                ch4_data)

    #avg_multi_spikes = avg_multiple_spikes(avg_ch_data,
    #                                    num_spikes)

    pca = PCA(n_components = 2)

    pca.fit(spike_data_4_channel)

    Y = pca.fit_transform(spike_data_4_channel)

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

    file = open('ground_noise_label.json')
    ground_noise_label = json.load(file)['Ground_Noise_Label']
    ground_noise_label = np.array(ground_noise_label)
    
    Y_ground_truth = Y[np.all([ ground_noise_label!=0,
                                ground_noise_label!=8,
                                ground_noise_label!=7], axis = 0)]
    Y_ground_label = ground_noise_label[np.all([
                                        ground_noise_label!=0,
                                        ground_noise_label!=8,
                                        ground_noise_label!=7], axis = 0)]

    Y_ground_truth = Y[ground_noise_label!=0]
    Y_ground_label = ground_noise_label[ground_noise_label!=0]

    groundtruth_iso_dist = isolation_distance(Y_ground_truth,Y_ground_label)
    print(groundtruth_iso_dist)
    #gm = GaussianMixture(n_components=8 , random_state=0).fit(Y_ground_truth)

    #labels = gm.predict(Y_ground_truth)

    clustering = MeanShift(bandwidth=50).fit(Y_ground_truth)
    labels = clustering.labels_

    print('Shape of spike data',np.shape(Y))

    acc_metrics = accuracy_metrics(labels, Y_ground_label,max(ground_truth_time[:num_spikes]))
    print('Accuracy of the model:',acc_metrics)

    prediction_iso_dist = isolation_distance(Y_ground_truth,labels)
    print('Isolation Distance:',prediction_iso_dist)

    avg_acc = 0
    for unit_acc in acc_metrics:

        avg_acc += acc_metrics[unit_acc]['acc']

    avg_acc /= len(acc_metrics)

    print('Avg Accuracy:', avg_acc)

    acc_list = []
    spike_rate_list = []
    iso_dist_list = []
    for unit_acc in acc_metrics:

        print(unit_acc,':',acc_metrics[unit_acc])

        acc_list.append(acc_metrics[unit_acc]['acc'])
        spike_rate_list.append(acc_metrics[unit_acc]['spike_rate'])
        iso_dist_list.append(groundtruth_iso_dist[unit_acc]['iso_dist'])


    plt.subplot(2,1,1)
    plt.scatter(Y_ground_truth[:,0], Y_ground_truth[:,1], c = labels)#facecolors='none', edgecolors=labels)
    plt.title('Prediction')
    plt.subplot(2,1,2)
    plt.scatter(Y_ground_truth[:,0], Y_ground_truth[:,1], c = Y_ground_label)#facecolors='none', edgecolors=ground_noise_label)#c = ground_noise_label )
    plt.title('Ground Truth')
    plt.show()

    plt.scatter(iso_dist_list, acc_list)
    plt.xlabel('Isolation Distance')
    plt.ylabel('Accuracy')
    plt.show()

    #plt.plot(ch3_data[:64])
    #plt.show()