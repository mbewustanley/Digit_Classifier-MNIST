import pandas as pd
import numpy as np
import seaborn as sns
import keras
import tensorflow as tf

from src.deepMl_04 import (
    load_data,
    reshape_and_encode,
    evaluate_cnn_model,
    add_result_to_json
)


# Load data
X_train_cnn, X_val_cnn, y_train, y_val = load_data()

# reshape and encode input data
X_train_cnn_4d, X_val_cnn_4d, y_train_ohe, y_val_ohe = reshape_and_encode()

# load model
#to later load model, use keras.saving.load_model('artifacts/DeepML/cnn_model.keras')
cnn_model = keras.saving.load_model('artifacts/DeepML/cnn_model.keras')
print(cnn_model.summary())

# evaluate cnn model
cnn_loss, cnn_accuracy = evaluate_cnn_model(cnn_model, X_val_cnn_4d, y_val_ohe)

# update results
#add_result_to_json(cnn_accuracy)  # fix to not completely modify the existing data. just add to it