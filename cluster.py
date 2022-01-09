from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift

class Cluster:

    '''
    '''

    def __init__(self):

        pass

    def meanshift_model(self,
                        X,
                        estimated_cluster_size=100):

        '''
        '''

        self.__meanshiftmodel = MeanShift(bandwidth=estimated_cluster_size).fit(X)

        return (self.__meanshiftmodel, 
                self.__meanshiftmodel.labels_)

    def meanshift_prediction(self,
                             X):

        '''
        '''

        return self.__meanshiftmodel.predict(X)

if __name__ == '__main__':

    a = 1
