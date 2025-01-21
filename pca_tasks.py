# -*- coding: utf-8 -*-
"""PCA-Tasks.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1h2-tCdSNTAjrSKLhT82pfORdM3Gfyhu-
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from google.colab import drive
drive.mount('/content/drive')

# import csv file as data
result_dir=r"/content/drive/MyDrive/Machine Learning/PCA-Tasks.csv"
df = pd.read_csv(result_dir)
# remove rows with Nan values
df = df.dropna()
# print the number of rows in df
print("Number of rows in df:", len(df))
# print the number of columns in df
print("Number of columns in df:", len(df.columns))
# inverse rows sot and IVTT
max_ivtt = df['winsorized_IVTT'].max()
# add a column that inverses IVTT values using max_ivtt
df['Inverse_IVTT_winsorized'] = max_ivtt - df['winsorized_IVTT']
max_sot = df['winsorized_SOT'].max()
# add a column that inverses SOT values using max_ivtt
df['Inverse_SOT_winsorized'] = max_sot - df['winsorized_SOT']
# remove df["winsorized_IVTT"] and df ['winsorized_SOT']
df = df.drop(columns=['winsorized_IVTT', 'winsorized_SOT'])
print(df.head())
# convert dataframe to dictionary
data = df.to_dict()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Perform PCA
pca = PCA(n_components=0.9)  # Keep all components initially
pca_result = pca.fit_transform(scaled_data)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Print results
print("Explained Variance Ratio (per component):", explained_variance)
print("\nCumulative Explained Variance:", np.cumsum(explained_variance))

# Loadings (contributions of each task to each component)
loadings = pd.DataFrame(pca.components_, columns=df.columns, index=[f'PC{i+1}' for i in range(len(pca.components_))])
print("\nPCA Loadings (Task Contributions):")
print(loadings)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center', label='Individual Explained Variance')
#plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='Cumulative Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.legend(loc='best')
plt.title('PCA Explained Variance')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', color='b', label='Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.xticks(range(1, len(explained_variance) + 1))  # Set x-ticks to 1, 2, 3, etc.
plt.title('PCA Scree Plot')
plt.legend(loc='best')
plt.show()

# Plot the loadings of each feature for the first few Principal Components
loadings.columns = [col.replace('winsorized_', '') for col in loadings.columns]
loadings.columns = [col.replace('_winsorized', '') for col in loadings.columns]
plt.figure(figsize=(12, 8))

# Select the top PCs to visualize
num_pcs_to_plot = min(5, len(loadings))  # Plot up to 5 PCs or as many as available

for i in range(num_pcs_to_plot):
    plt.bar(loadings.columns, loadings.iloc[i], alpha=0.7, label=f'PC{i+1}')

plt.xlabel('Features', fontsize=18)
plt.ylabel('Loadings (Contributions)', fontsize=18)
plt.title('Feature Contributions to Principal Components', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=14)  # Rotate feature names for better readability
plt.legend(loc='best')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Plot the PCA loadings heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(loadings, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label': 'Contribution'})
plt.title("PCA Loadings Heatmap", fontsize=14)
plt.xlabel("Features", fontsize=14)
plt.ylabel("Principal Components", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=14)  # Rotate feature names for better readability
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

from math import pi

# Data for radar chart (using PC1 as an example)
categories = loadings.columns.tolist()
values = loadings.iloc[0].values  # PC1 contributions
values = np.append(values, values[0])  # Close the circle

# Set up radar chart
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='b', alpha=0.25)
ax.plot(angles, values, color='b', linewidth=2)
ax.set_yticks([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=18, rotation=45, ha='right')
ax.set_title("Feature Contributions to PC1", fontsize=18)
plt.tight_layout()
plt.show()

import plotly.express as px
import pandas as pd

# Prepare PCA loading data (example for PC1)
categories = loadings.columns.tolist()  # Feature names (theta)
values = loadings.iloc[4].values  # PCA contributions for PC1 (r)

# Create a DataFrame for the radar chart
df = pd.DataFrame(dict(
    r=values,
    theta=categories
))

# Create the radar chart
fig = px.line_polar(df, r='r', theta='theta', line_close=True)

# Update trace to fill the area
fig.update_traces(fill='toself')

# Update layout for better readability
fig.update_layout(
    title="Feature Contributions to PC1",
    polar=dict(
        radialaxis=dict(visible=True, range=[0, max(values)*1.1]),  # Adjust range to fit data
    ),
    font=dict(size=16)
)

fig.show()

import plotly.express as px
import pandas as pd

# Prepare PCA loading data (example for PC1)
categories = loadings.columns.tolist()  # Feature names (theta)
values = loadings.iloc[1].values  # PCA contributions for PC1 (r)

# Create a DataFrame for the radar chart
df = pd.DataFrame(dict(
    r=values,
    theta=categories
))

# Create the radar chart
fig = px.line_polar(df, r='r', theta='theta', line_close=True)

# Update trace to fill the area
fig.update_traces(fill='toself')

# Update layout for better readability
fig.update_layout(
    title="Feature Contributions to PC1",
    polar=dict(
        radialaxis=dict(visible=True, range=[0, max(values)*1.1]),  # Adjust range to fit data
    ),
    font=dict(size=16)
)

fig.show()

plt.figure(figsize=(8, 6))
plt.scatter(loadings.iloc[0, :], loadings.iloc[1, :], c='b', label="Features")
for i, feature in enumerate(loadings.columns):
    plt.text(loadings.iloc[0, i] + 0.02, loadings.iloc[1, i], feature, fontsize=10)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel("Contribution to PC1")
plt.ylabel("Contribution to PC2")
plt.title("Feature Contributions to PC1 and PC2")
plt.grid()
plt.tight_layout()
plt.show()

# Calculate contributions for a specific data point
point = pca_result[0]  # Choose the first data point (or any index)
contributions = point[:, None] * loadings.values  # Multiply scores by loadings

# Rank variables by contribution to PC1 and PC2
pc1_contributions = contributions[:, 0]
pc2_contributions = contributions[:, 1]
print("Top contributors to PC1:", loadings.columns[np.argsort(-abs(pc1_contributions))])
print("Top contributors to PC2:", loadings.columns[np.argsort(-abs(pc2_contributions))])

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from google.colab import drive
drive.mount('/content/drive')

# import csv file as data
result_dir=r"/content/drive/MyDrive/Machine Learning/PCA-Tasks-without-corsi.csv"
df = pd.read_csv(result_dir)
# remove rows with Nan values
df = df.dropna()
# print the number of rows in df
print("Number of rows in df:", len(df))
# print the number of columns in df
print("Number of columns in df:", len(df.columns))
max_ivtt = df['winsorized_IVTT'].max()
# add a column that inverses IVTT values using max_ivtt
df['Inverse_IVTT_winsorized'] = max_ivtt - df['winsorized_IVTT']
max_sot = df['winsorized_SOT'].max()
# add a column that inverses SOT values using max_ivtt
df['Inverse_SOT_winsorized'] = max_sot - df['winsorized_SOT']
# remove df["winsorized_IVTT"] and df ['winsorized_SOT']
df = df.drop(columns=['winsorized_IVTT', 'winsorized_SOT'])
print(df.head())
# convert dataframe to dictionary
data = df.to_dict()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Perform PCA
pca = PCA(n_components=0.9)  # Keep all components initially
pca_result = pca.fit_transform(scaled_data)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance)

# Print results
print("Explained Variance Ratio (per component):", explained_variance)
print("\nCumulative Explained Variance:", np.cumsum(explained_variance))

# Loadings (contributions of each task to each component)
loadings = pd.DataFrame(pca.components_, columns=df.columns, index=[f'PC{i+1}' for i in range(len(pca.components_))])
print("\nPCA Loadings (Task Contributions):")
print(loadings)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center', label='Individual Explained Variance')
#plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='Cumulative Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.legend(loc='best')
plt.title('PCA Explained Variance')
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', color='b', label='Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.xticks(range(1, len(explained_variance) + 1))  # Set x-ticks to 1, 2, 3, etc.
plt.title('PCA Scree Plot')
plt.legend(loc='best')
plt.show()