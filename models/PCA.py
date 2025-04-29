from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Global objects to store state
_scaler = None
_pca = None

def fit_pca(feature_dict, n_components=10):
    global _scaler, _pca
    feature_matrix = _dict_to_matrix(feature_dict)
    
    _scaler = StandardScaler()
    scaled = _scaler.fit_transform(feature_matrix)

    _pca = PCA(n_components=n_components)
    reduced = _pca.fit_transform(scaled)
    
    return reduced

def transform_pca(feature_dict):
    global _scaler, _pca
    if _scaler is None or _pca is None:
        raise RuntimeError("Must call fit_pca before transform_pca.")
    
    feature_matrix = _dict_to_matrix(feature_dict)
    scaled = _scaler.transform(feature_matrix)
    return _pca.transform(scaled)

def get_explained_variance():
    global _pca
    if _pca is None:
        raise RuntimeError("PCA model has not been fitted yet.")
    return _pca.explained_variance_ratio_

def _dict_to_matrix(feature_dict):
    return np.array([list(features.values()) for features in feature_dict.values()])
