import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


train_df = pd.read_csv("data/train.csv")

# Image processing

def preprocess_images(df):
    """
    Preprocess images by normalizing pixel values to [0, 1].
    """
    # Separate features and labels
    pixel_cols = [col for col in df.columns if col != 'label']
    X_raw = df[pixel_cols].values.astype(np.float32) 
    y = df['label'].values.astype(np.int64)

    print(f'Raw feature matrix : {X_raw.shape}')
    print(f'Label array        : {y.shape}')
    print(f'Pixel value range  : [{X_raw.min()}, {X_raw.max()}]')

    # Normalize pixel values to [0, 1]
    X_normalized = X_raw / 255.0

    print(f'Normalized feature matrix : {X_normalized.shape}')
    print(f'Normalized pixel value range  : [{X_normalized.min()}, {X_normalized.max()}]')

    return X_normalized, y

X_normalized, y = preprocess_images(train_df)


# Standardize pixel values to have zero mean and unit variance
def standardize_images(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f'Standardized feature matrix : {X_scaled.shape}. mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}')
    return X_scaled

def apply_pca(X, n_components=100, random_state=42):
    """
    Apply PCA to reduce dimensionality while retaining 95% of variance.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_.cumsum()[-1]*100
    print(f'PCA reduced feature matrix : {X_pca.shape}. Explained variance ratio: {explained:.2f}%  ')
    
    # plot explained variance ratio
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100,
            color='#e74c3c', linewidth=2)
    plt.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90% threshold')
    plt.axvline(x=100, color='#3498db', linestyle='--', alpha=0.7, label='100 components')
    plt.title('PCA — Cumulative Explained Variance', fontsize=13, fontweight='bold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/pca_variance.png', dpi=150, bbox_inches='tight')
    #plt.show()
    print('✅ Saved: pca_variance.png')


    return X_pca, pca

# standardize and apply PCA
X_scaled = standardize_images(X_normalized)
X_pca, pca = apply_pca(X_scaled, n_components=100)



# split and save the processed data for classical models and cnn models
def split_n_save(X_pca, y, random_state=42):
    """
    split and Save the processed data to CSV files.
    """
    # Split the data into training and validation sets for classical models
    X_train_ml, X_val_ml, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=random_state, stratify=y)

    # split the data into training and validation sets for CNN models
    X_train_cnn, X_val_cnn, _ , _ = train_test_split(X_normalized, y, test_size=0.2, random_state=random_state, stratify=y)  

    # save the processed data to CSV files
    processed_data_dir = Path('data/processed')
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_data_dir / 'X_train_ml.npy', X_train_ml)
    np.save(processed_data_dir / 'X_val_ml.npy', X_val_ml)
    np.save(processed_data_dir / 'y_train.npy', y_train)
    np.save(processed_data_dir / 'y_val.npy', y_val)
    np.save(processed_data_dir / 'X_train_cnn.npy', X_train_cnn)
    np.save(processed_data_dir / 'X_val_cnn.npy', X_val_cnn)


    print('=' * 48)
    print('TRAIN / VALIDATION SPLIT SUMMARY')
    print('=' * 48)
    print(f'Training   samples (ML)  : {X_train_ml.shape[0]:,}')
    print(f'Validation samples (ML)  : {X_val_ml.shape[0]:,}')
    print(f'Feature dims (PCA)       : {X_train_ml.shape[1]}')
    print(f'\nTraining   samples (CNN) : {X_train_cnn.shape[0]:,}')
    print(f'Validation samples (CNN) : {X_val_cnn.shape[0]:,}')
    print(f'✅ Saved processed data to {processed_data_dir}')


split_n_save(X_pca, y, random_state=42)