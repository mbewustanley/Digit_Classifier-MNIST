import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import os
import json
from pathlib import Path

# Create artifacts directory if it doesn't exist
artifacts_dir = Path('artifacts')
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Load dataset --- can probably get it from kaggle but i downloaded it manually
train_df = pd.read_csv('data/train.csv')


def get_basic_info(df):
    """ Get basic information about the dataset. """
    
    # Safely convert pandas data types to string names
    dtype_counts = {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()}
    
    info = {
        'DATASET OVERVIEW': 'Basic information about the dataset',
        'Shape': list(df.shape),                          # Convert tuple to list
        'Rows (samples)': int(df.shape[0]),               # Cast to standard Python int
        'Columns': int(df.shape[1]),                     # Cast to standard Python int
        'Missing values': int(df.isnull().sum().sum()),   # Cast numpy.int64 to native int
        'Data types': dtype_counts                       # Cleaned dictionary
    }
    
    # Ensure the directory exists to avoid FileNotFoundError
    os.makedirs('artifacts', exist_ok=True)
    
    with open('artifacts/basic_info.json', 'w') as f:
        json.dump(info, f, indent=4)
        
    return info

get_basic_info(train_df)



def visualize_sample_images(df, num_samples=10):
    """
    Visualize sample images from the dataset.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        img = df.iloc[i, 1:].values.reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {df.iloc[i, 0]}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('artifacts/sample_images.png', dpi=150, bbox_inches='tight')
    #plt.show()
    print('✅ Saved: sample_images.png')

visualize_sample_images(train_df, num_samples=10)



def visualize_sample_images_for_each_digit(train_df):
    """
    Visualize ten sample image for each digit (0-9).
    """
    fig, axes = plt.subplots(10, 10, figsize=(15, 3))
    fig.suptitle('Sample Images for Each Digit (0-9)', fontsize=16, fontweight='bold')

    for digit in range(10):
        # Sample 10 random images of this digit
        samples = train_df[train_df['label'] == digit].sample(10, random_state=42)
        for j, (_, row) in enumerate(samples.iterrows()):
            img = row.drop('label').values.reshape(28, 28).astype(np.uint8)
            axes[digit][j].imshow(img, cmap='gray')
            axes[digit][j].axis('off')
            if j == 0:
                axes[digit][j].set_ylabel(f'  {digit}', fontsize=12,
                                        fontweight='bold', rotation=0, labelpad=20)
                axes[digit][j].yaxis.set_label_position('left')

    plt.tight_layout()
    plt.savefig('artifacts/sample_images_each_digit.png', dpi=150, bbox_inches='tight')
    #plt.show()
    print("="*50)
    print('✅ Saved: sample_images_each_digit.png')

visualize_sample_images_for_each_digit(train_df)


# Visualize the distribution of labels

def plot_class_distribution(train_df):
    """
    Plot the distribution of classes in the dataset.
    """
    label_counts = train_df['label'].value_counts().sort_index()

    # Save class distribution information to a JSON file
        # Clean up Class Distribution dictionary keys and values
    clean_distribution = {str(k): int(v) for k, v in label_counts.to_dict().items()}

        # Clean up the unique classes list
    clean_classes = [int(x) if hasattr(x, 'item') else x for x in sorted(train_df['label'].unique())]

    # Save class distribution information to a JSON file
    label_distribution = {
        'Class Distribution': clean_distribution,
        'Total Samples': int(len(train_df)),
        'Classes': clean_classes,
        'Balance Check': {
            'Min': int(label_counts.min()),
            'Max': int(label_counts.max()),
            'Std': float(label_counts.std())  # Converts numpy.float64 to python float
        }
    }

    os.makedirs('artifacts', exist_ok=True)

    with open('artifacts/class_distribution.json', 'w') as f:
        json.dump(label_distribution, f, indent=4)



    # ------------------------------------------------------------------------------------
    # Create bar chart and pie chart for class distribution

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    bars = axes[0].bar(label_counts.index, label_counts.values,
                    color=colors, edgecolor='black', width=0.7)
    axes[0].set_title('Digit Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Digit', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xticks(range(10))
    axes[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, label_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                    str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Pie chart
    axes[1].pie(label_counts.values,
                labels=[f'Digit {i}' for i in range(10)],
                autopct='%1.1f%%', colors=colors,
                startangle=90, pctdistance=0.8)
    axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('artifacts/class_distribution.png', dpi=150, bbox_inches='tight')
    #plt.show()
    print('✅ Saved: class_distribution.png')

plot_class_distribution(train_df)


# pixel intensity distribution for each digit
def plot_pixel_intensity_distribution(train_df):
    """
    Plot the pixel intensity distribution for each digit.
    """
    pixel_cols = [col for col in train_df.columns if col != 'label']
    pixel_vals = train_df[pixel_cols].values.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of pixel intensities
    axes[0].hist(pixel_vals, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0].set_title('Pixel Intensity Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Pixel Value (0–255)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(alpha=0.3)

    # Average image per digit
    avg_images = []
    for d in range(10):
        avg = train_df[train_df['label'] == d][pixel_cols].mean().values.reshape(28, 28)
        avg_images.append(avg)

    axes[1].set_title('Average Image per Digit', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    sub_axes = [axes[1].inset_axes([0.1*i, 0, 0.1, 1]) for i in range(10)]
    for i, (ax, avg) in enumerate(zip(sub_axes, avg_images)):
        ax.imshow(avg, cmap='hot')
        ax.set_title(str(i), fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('artifacts/pixel_intensity_distribution.png', dpi=150, bbox_inches='tight')
    #plt.show()
    print('✅ Saved: pixel_intensity_distribution.png')

plot_pixel_intensity_distribution(train_df)