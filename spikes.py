import json

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

class SpikeTime:

    def __init__(self,
                 time_duration,
                 processing_time_duration,
                 sampling_frequency,
                 spikes_detection_time=None,
                 spikes_true_time=None
                 ):

        # Time duration to be considered

        self.time_duration = time_duration
        
        # Time duration for 1 processing chunk

        self.processing_time_duration = processing_time_duration
        self.sampling_freq = sampling_frequency

        # Number of samples in total time duration

        self.num_samples = int(self.time_duration*3600*self.sampling_freq)
        
        # Number of samples for 1 chunk

        self.num_samples_processing = int(self.processing_time_duration*3600*self.sampling_freq)

        self.spikes_occurence_time = spikes_detection_time

        # Ground truth time for true spikes

        self.spikes_true_time = spikes_true_time

    def no_of_spikes(self):

        '''
        '''

        # Number of spikes corresponding to given time duration

        self.num_spikes = len(self.spikes_occurence_time)

        return self.num_spikes



class SpikeWave:

    def __init__(self,
                 spikes_waveform,
                 feature_scaling=10):
        
        self.spikes_waveform = spikes_waveform/feature_scaling
        

class SpikeLabel:

    def __init__(self,
                 labels_without_noise=None,
                 labels_with_noise=None):

        # Ground truth labels for true spikes

        self.true_spikes_labels = labels_without_noise


        self.true_noise_spikes_labels = labels_with_noise

    def jsonify_true_noise_labels(self,
                                  num_spikes,
                                  spikes_detection_time,
                                  spikes_true_time):

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

            simulation_time = spikes_detection_time[i]

            # Default label for current spike as true
            # if it does not gets any other label

            spike_is_noise = True
            
            # Counting number of iterations completed
            # for searching spike in ground_truth spike
            # time

            j = 0

            # Iterating over ground truth spike times
            
            for real_time in spikes_true_time[idx_readed:]:

                # Count increases by 1 as one iteration for seaching spike 
                # started

                j += 1

                # If difference of samples between simulation and real time is 
                # less then 2 samples, then spikes will be considered as true
                # spike 

                if np.abs(simulation_time-real_time) < 2:

                    # As simulatio spike has been detected as true spike
                    # it should be given label from the ground truth 
                    
                    ground_noise_label.append(int(self.true_spikes_labels[spikes_true_time.index(real_time)]))
                    
                    # Update the index upto which spike time has already been
                    # checked
                    
                    idx_readed = spikes_true_time.index(real_time)
                    
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
            pass
            
