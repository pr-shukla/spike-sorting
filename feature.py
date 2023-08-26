from sklearn.decomposition import PCA

class FeatureSelection:

    '''
    Provides different implementation of dimension reduction
    '''

    def __init__(self) -> None:
        pass

    def pca_model(self,
                  X,
                  num_pca_components=8):

        '''
        Applies PCA technique for dimension reduction

        Parameters
        ----------
        X: np.array
            n-dim input data of spikes
        
        num_pca_components: int
            Desired dimennsion of data

        Return
        ------
        self.__pca_model: PCA
            PCA model to find principal components
        '''

        self.__pca_model = PCA(n_components = num_pca_components).fit(X)

        return self.__pca_model

    def pca_features(self,
                     X):

        '''
        Applies fitted PCA model  on spike data
        '''

        return self.__pca_model.fit_transform(X)
