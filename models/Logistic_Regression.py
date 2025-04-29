import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def run_logistic_regression(all_features: dict, labels: dict, test_size: float = 0.2, random_state: int = 42):

    X = [] # feature matrix X
    y = [] # label vector y 

    for chan, features in all_features.items():
        if chan in labels:
            X.append(list(features.values()))
            y.append(labels[chan])

    if not X or not y:
        raise ValueError("No matching features and labels found. Check your inputs.")

    X = np.array(X)
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    # Print results
    print("\n=== Logistic Regression Classification Report ===")
    print(report)

    return model, report

def predict_logistic_regression(model, all_features):

    # Convert the feature dictionaries into a matrix (X)
    feature_keys = list(next(iter(all_features.values())).keys())
    X = np.array([[features.get(key, 0) for key in feature_keys] for features in all_features.values()])

    # Predict probabilities
    y_pred = model.predict_proba(X)[:, 1]  # Probability of class "1"

    # Convert probabilities into labels
    y_pred_labels = (y_pred >= 0.5).astype(int)

    return y_pred, y_pred_labels
