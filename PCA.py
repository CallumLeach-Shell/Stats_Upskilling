class PCA():

    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        '''
        This fitting algorithm uses the covariance method.
        '''

        #Standarise the data
        X = X.copy()
        self.mean = sum(X)/len(X)
        self.std = (sum((i - self.mean)**2 for i in X)/len(X))**0.5
        X_std = (X - self.mean)/self.std

        # Define the covariance matrix
        cov = (X_std.T @ X_std)/(X_std.shape[0]-1)

        # Eigendecomposition of covariance matrix.
        eig_vals, eig_vecs = eig(cov)

        # Adjusting the eigenvectors (loadings) that are largest in absolute value to be positive
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
        signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
        eig_vecs = eig_vecs*signs[np.newaxis,:]
        eig_vecs = eig_vecs.T

        # Rearrange the eigenvectors and eigenvalues
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i,:]) for i in range(len(eig_vals))]

        #Sort this tuple list in descending order based on eigenvalues magnitude
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])

        self.components = eig_vecs_sorted[:self.n_components,:]

        # Explained variance ratio
        self.explained_variance_ratio = [i/np.sum(eig_vals) for i in eig_vals_sorted[:self.n_components]]
        
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

        return self

    def transform(self, X):
        X = X.copy()
        X_std = (X - self.mean) / self.std
        X_proj = X_std.dot(self.components.T)

        return X_proj


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import numpy as np
    from numpy.linalg import eig
    import matplotlib.pyplot as plt 

    iris = load_iris()
    X = iris['data']
    y = iris['target']

    PCA = PCA(n_components=2).fit(X)

    print('Components:\n', PCA.components)
    print('Explained variance ratio:\n', PCA.explained_variance_ratio)
    print('Cumulative explained variance:\n', PCA.cum_explained_variance)

    X_proj = PCA.transform(X)
    print('Transformed data shape:', X_proj.shape)

    plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y)
    plt.show()
