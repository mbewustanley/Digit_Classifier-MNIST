import os
import pandas as pd
import numpy as np
import json

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.classicUtils import train_and_evaluate_model, save_artifacts
from src.model_compareUtils import create_model_comparison_table, plot_confusion_matrix


# Model training and evaluation

# Load the processed data
X_train_ml = np.load('data/processed/X_train_ml.npy')
y_train_ml = np.load('data/processed/y_train.npy')
y_val_ml = np.load('data/processed/y_val.npy')
x_val_ml = np.load('data/processed/X_val_ml.npy')


# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
knn_results, knn_pred, knn_model = train_and_evaluate_model(knn_model, X_train_ml, y_train_ml, x_val_ml, y_val_ml, "k-nearest neighbors (KNN)")
save_artifacts(knn_results, y_val_ml, knn_pred, knn_model, "knn")


# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_results, lr_pred, lr_model = train_and_evaluate_model(lr_model, X_train_ml, y_train_ml, x_val_ml, y_val_ml, "logistic regression (LR)")
save_artifacts(lr_results, y_val_ml, lr_pred, lr_model, "lr")


# random forest (RF)    

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_results, rf_pred, rf_model = train_and_evaluate_model(rf_model, X_train_ml, y_train_ml, x_val_ml, y_val_ml, "random forest (RF)")
save_artifacts(rf_results, y_val_ml, rf_pred, rf_model, "rf")


# Support vector machine (SVM)
# RBF kernel SVM — best accuracy among classical methods for MNIST
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_results, svm_pred, svm_model = train_and_evaluate_model(svm_model, X_train_ml, y_train_ml, x_val_ml, y_val_ml, "support vector machine (SVM)")
save_artifacts(svm_results, y_val_ml, svm_pred, svm_model, "svm")


# compare models
results_df = create_model_comparison_table(knn_results, lr_results, rf_results, svm_results)
plot_confusion_matrix(y_val_ml, knn_pred, "knn")
# plot_confusion_matrix(y_val_ml, lr_pred, "lr")
# plot_confusion_matrix(y_val_ml, rf_pred, "rf")
# plot_confusion_matrix(y_val_ml, svm_pred, "svm")