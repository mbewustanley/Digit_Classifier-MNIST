import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib
from sklearn.metrics import accuracy_score, classification_report


# create a function to train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, name):
    # fit model, time it, and return evaluation metrics

    start_time = time.time()

    model.fit(X_train, y_train)

    end_time = time.time()
    training_time = end_time - start_time

    # run predictions
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"{name} - Accuracy: {acc:.4f}, Training time: {training_time:.2f} seconds")
    print("-" * 50)
    print(f"Classification Report for {name}:\n")
    print(classification_report(y_val, y_pred, target_names=[str(i) for i in range(10)]))

    return {
        "Model": name,
        "Accuracy": round(acc, 2),
        "Training Time": round(training_time, 2)
    }, y_pred, model

# save all artifacts.
def save_artifacts(results, y_val, y_pred, model, name):
    # save results to json
    os.makedirs(f'artifacts/evaluate/{name}', exist_ok=True)
    with open(f'artifacts/evaluate/{name}/{name}_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # save classification report to json
    report = classification_report(y_val, y_pred, target_names=[str(i) for i in range(10)], output_dict=True)
    with open(f'artifacts/evaluate/{name}/{name}_classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)

    # save model
    joblib.dump(model, f'artifacts/evaluate/{name}/{name}_model.pkl')


