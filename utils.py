import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

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

def spike_times(file, no_of_spikes, filename):

    '''
    '''

    spike_time_data = list(file.read(no_of_spikes))

    filepath = 'F:/Neuroscience/data/' + filename

    spike_time_data = np.fromfile(filepath, dtype=np.uint64,count=no_of_spikes)    

    return spike_time_data

def spike_data_from_channels(file, no_of_spikes, filename):

    '''
    '''

    no_of_samples = no_of_spikes*num_channels*num_samples

    filepath = 'F:/Neuroscience/data/' + filename

    data = list(file.read(no_of_samples))

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

if __name__ == '__main__':
    num_spikes = 1000

    spike_binary_file = read_binary_file('Spikes')

    ch1_data, ch2_data, ch3_data, ch4_data,spike_data_4_channel = spike_data_from_channels(spike_binary_file ,
                                                                                           num_spikes,
                                                                                           'Spikes')

    spike_time_binary_file = read_binary_file('SpikeTimes')

    spike_time_data = spike_times(spike_time_binary_file,
                                  num_spikes,
                                  'SpikeTimes')

    mat_data = read_mat_file('dataset_params')

    print('Parameters of dataset:', mat_data.keys())
    print('Parameter Units lenght', mat_data['sp_u'][0][0][0])
    print('Spike Times:', spike_time_data[:20])

    ground_truth = mat_data['sp_u'][0][0][0][:num_spikes]
    print(len(ch1_data), len(ch2_data), len(ch3_data), len(ch4_data))

    print('Shape of spike data',np.shape(spike_data_4_channel))

    avg_ch_data = avg_spikes_channels(ch1_data,
                                    ch2_data,
                                    ch3_data,
                                    ch4_data)

    avg_multi_spikes = avg_multiple_spikes(avg_ch_data,
                                        num_spikes)

    print(avg_ch_data[:20])

    pca = PCA(n_components = 2)

    pca.fit(spike_data_4_channel)

    Y = pca.fit_transform(spike_data_4_channel)
    gm = GaussianMixture(n_components=8 , random_state=0).fit(Y)

    labels = gm.predict(Y)

    print('Shape of spike data',np.shape(Y))

    plt.subplot(2,1,1)
    plt.scatter(Y[:,0], Y[:,1], c = labels)
    plt.subplot(2,1,2)
    plt.scatter(Y[:,0], Y[:,1], c = ground_truth )
    plt.show()

    #plt.plot(ch3_data[:64])
    #plt.show()