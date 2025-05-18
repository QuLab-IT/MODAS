import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

_scaler = None
_pca = None

def fit_pca(features, n_components=None, variance_threshold=None):
    """
    Fit PCA on features, either with a fixed number of components or
    choosing components to explain at least variance_threshold of variance.

    Args:
        features: 2D array-like, shape (n_samples, n_features)
        n_components: int or None - if None, use variance_threshold
        variance_threshold: float in (0,1), ignored if n_components is set

    Returns:
        reduced_features: PCA-transformed feature matrix
    """
    global _scaler, _pca
    feature_matrix = _prepare_matrix(features)

    _scaler = StandardScaler()
    scaled = _scaler.fit_transform(feature_matrix)

    if n_components is None:
        # Fit PCA with all components to compute explained variance
        pca_full = PCA()
        pca_full.fit(scaled)

        # Find number of components to explain desired variance
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.searchsorted(cum_var, variance_threshold) + 1

        print(f"Selected n_components={n_components} to reach {variance_threshold*100:.1f}% explained variance")

    _pca = PCA(n_components=n_components)
    reduced = _pca.fit_transform(scaled)

    return reduced

def transform_pca(features):
    global _scaler, _pca
    if _scaler is None or _pca is None:
        raise RuntimeError("Must call fit_pca before transform_pca.")

    feature_matrix = _prepare_matrix(features)
    scaled = _scaler.transform(feature_matrix)
    return _pca.transform(scaled)

def get_explained_variance():
    global _pca
    if _pca is None:
        raise RuntimeError("PCA model has not been fitted yet.")
    return _pca.explained_variance_ratio_

def plot_explained_variance():
    global _pca
    if _pca is None:
        raise RuntimeError("PCA model has not been fitted yet.")

    evr = _pca.explained_variance_ratio_
    cum_var = np.cumsum(evr)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(evr) + 1), evr, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1, len(evr) + 1), cum_var, where='mid',
             label='Cumulative explained variance')
    plt.xlabel('Principal component index')
    plt.ylabel('Explained variance ratio')
    plt.title('Explained Variance by PCA Components')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def _prepare_matrix(features):
    if isinstance(features, np.ndarray):
        if features.ndim != 2:
            raise ValueError("Input NumPy array must be 2D.")
        return features
    elif isinstance(features, pd.DataFrame):
        return features.to_numpy()
    elif isinstance(features, dict):
        matrix = []
        for item in features.values():
            if isinstance(item, dict):
                matrix.append(list(item.values()))
            elif isinstance(item, np.ndarray):
                matrix.append(item.tolist())
            else:
                raise TypeError(f"Unsupported feature type: {type(item)}")
        return np.array(matrix)
    else:
        raise TypeError(f"Unsupported input type: {type(features)}")
