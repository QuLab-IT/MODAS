from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Global objects to store state
_scaler = None
_pca = None

def fit_pca(features, n_components=10):
    global _scaler, _pca
    feature_matrix = _prepare_matrix(features)

    _scaler = StandardScaler()
    scaled = _scaler.fit_transform(feature_matrix)

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
