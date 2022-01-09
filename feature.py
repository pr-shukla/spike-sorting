from sklearn.decomposition import PCA

class FeatureSelection:

    '''
    '''

    def __init__(self) -> None:
        pass

    def pca_model(self,
                  X,
                  num_pca_components=8):

        '''
        '''

        self.__pca_model = PCA(n_components = num_pca_components).fit(X)

        return self.__pca_model

    def pca_features(self,
                     X):

        '''
        '''

        return self.__pca_model.fit_transform(X)