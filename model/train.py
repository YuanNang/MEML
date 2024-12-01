from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import numpy as np
from lightgbm import LGBMClassifier as LGBC
import os
import pandas as pd
import joblib

# Hyperparameters for each model
params_stability = {
 'n_estimators': 420,
 'learning_rate': 0.06879639007831825,
 'num_leaves': 80,
 'max_depth': 7,
 'min_data_in_leaf': 80,
 'lambda_l1': 0,
 'lambda_l2': 5,
 'min_gain_to_split': 0.005197984933632771,
 'bagging_fraction': 0.9,
 'bagging_freq': 1,
 'feature_fraction': 0.7,
 'verbose': -1,
 'random_state':1207
              }

params_semiconductor = {
 'n_estimators': 390,
 'learning_rate': 0.04198805200374064,
 'num_leaves': 45,
 'max_depth':15,
 'min_data_in_leaf': 40,
 'lambda_l1': 0,
 'lambda_l2': 0,
 'bagging_fraction': 1,
 'bagging_freq': 1,
 'feature_fraction': 1,
 'verbose': -1,
 'random_state':1207,
}

params_gap_type = {
 'n_estimators': 360,
 'learning_rate': 0.039456919445208266,
 'num_leaves': 36,
 'max_depth': 8,
 'min_data_in_leaf': 20,
 'lambda_l1': 0,
 'lambda_l2': 5,
 'bagging_fraction': 1,
 'bagging_freq': 1,
 'feature_fraction': 1,
 'verbose': -1,
 'is_unbalance':True,
 'random_state':1207
              }

def evaluate_classifier(X, y, classifier, best_params, n_splits=5, random_state=1207):
    """
    Evaluates a classifier using Stratified K-Folds cross-validation.

    Parameters:
    - X: Features DataFrame
    - y: Target variable
    - classifier: The classifier to be used (e.g., LGBC)
    - best_params: Dictionary of hyperparameters for the classifier
    - n_splits: Number of splits for Stratified K-Fold (default 5)
    - random_state: Random state for reproducibility (default 1207)
    """

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classifier.set_params(**best_params)

    # Initialize lists to store metrics
    f1_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []

    tprs = []
    true_fprs = []
    true_tprs = []

    mean_fpr = np.linspace(0, 1, 100)

    # Loop over each fold
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        classifier.fit(X.loc[train_idx], y[train_idx], categorical_feature=['spacegroup'])

        # Predict and compute the metrics
        y_pred = classifier.predict(X.loc[test_idx])

        f1_scores.append(f1_score(y[test_idx], y_pred, average='weighted'))
        precision_scores.append(precision_score(y[test_idx], y_pred, average='weighted'))
        recall_scores.append(recall_score(y[test_idx], y_pred, average='weighted'))

        # ROC and AUC
        fpr, tpr, _ = roc_curve(y[test_idx], classifier.predict_proba(X.loc[test_idx])[:, 1])
        auc_score = roc_auc_score(y[test_idx], classifier.predict_proba(X.loc[test_idx])[:, 1])
        auc_scores.append(auc_score)

        # Interpolate true positive rate to get smooth curves
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        tprs.append(interp_tpr)
        true_fprs.append(fpr)
        true_tprs.append(tpr)

    # Calculate the mean and standard deviation of AUCs and tprs
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    # Return the metrics in a dictionary
    # Print the metrics
    print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Precision Score: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall Score: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"AUC: {mean_auc:.4f} ± {std_auc:.4f}")

if __name__ == "__main__":
    data_path = os.path.join('..', 'data', 'data_all.xlsx')
    data = pd.read_excel(data_path)

    X = data.iloc[:, 6:]

    y_stability = (data['Decomposition_energy'] < 0.1).astype(int)   # Stability label
    y_semiconductor = (data['bandgap'] != 0).astype(int)  # Semiconductor label
    semiconductors = data[data['bandgap'] > 0].reset_index()  # Filter semiconductor data
    y_gap_type = (semiconductors['is_gap_direct'] == 1).astype(int) # Direct bandgap label

    print('model stability-----------------------------')
    evaluate_classifier(X, y_stability, classifier=LGBC(), best_params=params_stability)
    print('model semiconductor-----------------------------')
    evaluate_classifier(X, y_semiconductor, classifier=LGBC(), best_params=params_semiconductor)
    print('model gap type-----------------------------')
    evaluate_classifier(semiconductors.iloc[:,7:], y_gap_type, classifier=LGBC(), best_params=params_semiconductor)

    # Train and save final models
    model_stability = LGBC(**params_stability)
    model_stability = model_stability.fit(X, y_stability)
    joblib.dump(model_stability, 'stability_model.pkl')

    model_semiconductor = LGBC(**params_semiconductor)
    model_semiconductor = model_semiconductor.fit(X, y_semiconductor)
    joblib.dump(model_semiconductor, 'semiconductor_model.pkl')

    model_gap_type = LGBC(**params_gap_type)
    model_gap_type = model_gap_type.fit(semiconductors.iloc[:, 7:].values, y_gap_type)
    joblib.dump(model_gap_type, 'gap_type_model.pkl')

    print("Final models saved successfully.")