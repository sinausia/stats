'''
 I am using DTW and kmeans to cluster time series. The csv used here has observations as
 columns, time steps as rows. No header or anything, just numbers

'''


import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import pandas as pd
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples

num_samples = 3


def z_normalization(file_path, num_samples = num_samples, num_timesteps=910):
    data = pd.read_csv(file_path).iloc[:num_timesteps, :num_samples]
    X = data.values.T  # Transpose to get samples as rows and time steps as columns
    scaler = TimeSeriesScalerMeanVariance()
    X_normalized = scaler.fit_transform(X)
    return X_normalized


def max_min_normalization(file_path, num_samples=num_samples, num_timesteps=910):
    data = pd.read_csv(file_path).iloc[:num_timesteps, :num_samples]
    X = data.values.T  # Transpose to get samples as rows and time steps as columns
    X_normalized = X / np.max(np.abs(X), axis=1, keepdims=True)
    return X_normalized


def plot_normalized_samples(X_normalized, num_samples=num_samples):
    colors = cm.magma(np.linspace(0, 1, num_samples))
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(num_samples):
        x = np.arange(len(X_normalized[i])) * 1.1
        ax.plot(x, X_normalized[i].ravel(), label=f'Sample {i}', color=colors[i])  # Assign color from colormap
    if num_samples > 5 and num_samples < 15:
        ncol = 2
    elif num_samples > 14:
        ncol = 3
    else:
        ncol=1
    ax.legend(fontsize=14, loc='lower right', bbox_to_anchor=(1.0, 0.0), ncol=ncol)
    #ax.set_title('Normalized Time Series Samples', fontsize=18)
    ax.set_xlim(0, np.max([len(sample) * 1.1 for sample in X_normalized]))
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Normalized scores', fontsize=14)
    plt.show()
    
    
def elbow_method(X_normalized, k_range=range(1, num_samples - 1)):
    sum_of_squared_distances = []
    for k in k_range:
        km = TimeSeriesKMeans(n_clusters=k, n_init=2, metric="dtw", verbose=False, max_iter_barycenter=10, random_state=0) #n_init is the number of initial cluster centroids selected to find the optimal
        km = km.fit(X_normalized)
        sum_of_squared_distances.append(km.inertia_) # inertia = sum of squared distances of samples to their closest cluster center
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_range, sum_of_squared_distances, 'bx-')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Sum of squared distances')
    ax.set_title('Elbow Method For Optimal k')
    plt.show()

def silhouette_analysis(X_normalized, k_range=range(2, num_samples - 1)):
    X_reshaped = X_normalized.reshape(X_normalized.shape[0], -1)
    if not k_range:
        print('Silhouette analysis couldn´t be performed, k_range is empty (num_samples = initial value of k_range)')
    else:
        for n_clusters in k_range:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
    
            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X_normalized) + (n_clusters + 1) * 10])
    
            clusterer = TimeSeriesKMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = clusterer.fit_predict(X_normalized)
    
            silhouette_avg = silhouette_score(X_normalized, cluster_labels, metric="dtw") # the average score
            print(f"For n_clusters = {n_clusters}, the average silhouette_score is: {silhouette_avg:.2f}")
    
            sample_silhouette_values = silhouette_samples(X_reshaped, cluster_labels) # the individual scores of every sample
    
            y_lower = 10
            for i in range(n_clusters):  # to plot the individual scores for different number of clusters
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i] # extract the scores per sample
                ith_cluster_silhouette_values.sort()
    
                size_cluster_i = ith_cluster_silhouette_values.shape[0] # calculate the number of samples in the cluster
                y_upper = y_lower + size_cluster_i
    
                color = cm.magma(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
                y_lower = y_upper + 10
    
            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
    
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax1.set_yticks([]) 
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
            colors = cm.magma(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X_reshaped[:, 0], X_reshaped[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    
            # Plot cluster centers
            centers = clusterer.cluster_centers_.reshape(n_clusters, -1)
            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="o", c="white", alpha=1, s=200, edgecolor="k")
                ax2.scatter(c[0], c[1], marker=f"${i}$", alpha=1, s=50, edgecolor="k")
    
            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")
    
            plt.suptitle(f"Silhouette analysis for KMeans clustering on time series data with n_clusters = {n_clusters}", fontsize=14, fontweight="bold")
    
        plt.show()

def cluster_and_visualize(X_normalized, n_clusters=2): # n_clusters=2 is just a default, can change it later
    dba_km = TimeSeriesKMeans(n_clusters=n_clusters, n_init=2, metric="dtw", verbose=False, max_iter_barycenter=10, random_state=0)
    y_pred_dba_km = dba_km.fit_predict(X_normalized)

    silhouette_avg = silhouette_score(X_normalized, y_pred_dba_km, metric="dtw")
    print(f"DBA silhouette score: {silhouette_avg:.2f}")

    plt.figure(figsize=(15, 15))
    sz = X_normalized.shape[1]
    for yi in range(n_clusters):
        plt.subplot(3, n_clusters, yi + 1)
        for xx in X_normalized[y_pred_dba_km == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        #plt.ylim(-1, 1)
        plt.text(0.55, 0.85, f'Cluster {yi + 1}', transform=plt.gca().transAxes)
        if yi == n_clusters - 1:
            plt.title("DBA $k$-means")

    plt.tight_layout()
    plt.show()
    
    for i, label in enumerate(y_pred_dba_km):
        print(f"Column {i} is in cluster {label}")

file_path = "/Users/danielsinausia/Documents/test/PCA_scores.csv"
X_normalized = z_normalization(file_path)
#X_normalized = max_min_normalization(file_path)
plot_normalized_samples(X_normalized)
elbow_method(X_normalized)
silhouette_analysis(X_normalized)
cluster_and_visualize(X_normalized, n_clusters=2)
