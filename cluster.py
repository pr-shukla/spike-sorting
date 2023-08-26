from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift

class Cluster:

    '''
    Interface for different clustering algorithms
    '''

    def __init__(self):

        pass

    def meanshift_model(self,
                        X,
                        estimated_cluster_size=100):

        '''
        Implements Meanshift Algorithm on given data

        Parameters
        ----------
        X: np.array
            Points of each spike in n-dim
            
        estimated_cluster_size: int
            Approx how many clsuters could be there

        Returns
        -------
        self.__meanshiftmodel: MeanShift
            Clustering model fitted on given data
            
        self.__meanshiftmodel.labels_: np.array
            Each data point labeled with cluster to which it belongs in order
        '''

        self.__meanshiftmodel = MeanShift(bandwidth=estimated_cluster_size).fit(X)

        return (self.__meanshiftmodel, 
                self.__meanshiftmodel.labels_)

    def meanshift_prediction(self,
                             X):

        '''
        Cluster label prediction for each datapoint
        '''

        return self.__meanshiftmodel.predict(X)

if __name__ == '__main__':

    a = 1
