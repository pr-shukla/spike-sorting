import numpy as np
import matplotlib.pyplot as plt

from utils import read_mat_file, read_binary_file, spike_data_from_channels, avg_spikes_channels, avg_multiple_spikes

num_spikes = 10000

spike_binary_file = read_binary_file('Spikes')

ch1_data, ch2_data, ch3_data, ch4_data,spike_data_4_channel = spike_data_from_channels(spike_binary_file ,
                                                                                       num_spikes,
                                                                                       'Spikes')

spike_time_binary_file = read_binary_file('SpikeTimes')

mat_data = read_mat_file('dataset_params')

ground_truth = np.array(mat_data['sp_u'][0][0][0][:num_spikes])

#print((spike_data_4_channel[1]))
unit_data = spike_data_4_channel[ground_truth==1]

avg_unit = np.sum(unit_data, axis=0)/num_spikes

plt.subplot(2,2,1)
plt.title('Channel 1')
plt.plot(avg_unit[:64])
plt.subplot(2,2,2)
plt.title('Channel 2')
plt.plot(avg_unit[64:128])
plt.subplot(2,2,3)
plt.title('Channel 3')
plt.plot(avg_unit[128:192])
plt.subplot(2,2,4)
plt.title('Channel 4')
plt.plot(avg_unit[192:256])

plt.show()