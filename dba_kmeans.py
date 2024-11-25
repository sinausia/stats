
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import pandas as pd
import matplotlib.cm as cm
import os
from sklearn.metrics import davies_bouldin_score


row_range = (80, 180)
include_folders = {
    "DS_00132", "DS_00133", "DS_00134", "DS_00127", "DS_00163", "DS_00131",
    "DS_00138", "DS_00135", "DS_00139", "DS_00136", "DS_00140", "DS_00137",
    "DS_00141", "DS_00144", "DS_00142", "DS_00145", "DS_00143", "DS_00146",
    "DS_00181", "DS_00180", "DS_00148", "DS_00152", "DS_00149", "DS_00153",
}
ignore_folders = {
    "raw data", "CO peak", "Stark shift", "2500_to_3999", "1001_to_3999",
    "1635_peak", "2000_to_3999", "900_to_3999", "650_to_4000", "Diffusion coefficient plots",
    "DS_00145_01", "First derivative", "non-mean-centered"
}
base_dir = "...t"


def z_normalization(data):
    scaler = TimeSeriesScalerMeanVariance()
    return scaler.fit_transform(data)

def plot_normalized_samples(X_normalized, output_folder):
    num_samples = X_normalized.shape[0]
    colors = cm.magma(np.linspace(0, 1, num_samples))
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        x = np.arange(len(X_normalized[i])) * 1.1
        plt.plot(x, X_normalized[i].ravel(), label=f'Sample {i}', color=colors[i])
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Normalized scores', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_folder, "normalized_samples_combined.png"))
    plt.close()

def elbow_method(X_normalized, output_folder):
    k_range = range(1, min(10, X_normalized.shape[0]))
    sum_of_squared_distances = []
    for k in k_range:
        km = TimeSeriesKMeans(n_clusters=k, n_init=2, metric="dtw", max_iter_barycenter=10, random_state=0)
        km = km.fit(X_normalized)
        sum_of_squared_distances.append(km.inertia_)
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, sum_of_squared_distances, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method For Optimal k')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "elbow_method_combined.png"))
    plt.close()

def silhouette_analysis(X_normalized):
    k_range = range(2, min(10, X_normalized.shape[0]))
    silhouette_scores = []
    for n_clusters in k_range:
        clusterer = TimeSeriesKMeans(n_clusters=n_clusters, random_state=0, metric="dtw")
        cluster_labels = clusterer.fit_predict(X_normalized)
        silhouette_avg = silhouette_score(X_normalized, cluster_labels, metric="dtw")
        silhouette_scores.append((n_clusters, silhouette_avg))
    best_k = max(silhouette_scores, key=lambda x: x[1])[0]
    print(f"Best number of clusters (k): {best_k}")
    return best_k

def cluster_and_visualize(X_normalized, n_clusters, output_folder, sample_info):
    dba_km = TimeSeriesKMeans(n_clusters=n_clusters, n_init=2, metric="dtw", max_iter_barycenter=10, random_state=0)
    y_pred = dba_km.fit_predict(X_normalized)
    for i, cluster in enumerate(y_pred):
        sample_info[i]["Cluster"] = int(cluster + 1)
    pd.DataFrame(sample_info).to_csv(os.path.join(output_folder, "cluster_assignments_combined.csv"), index=False)
    for yi in range(n_clusters):
        plt.figure(figsize=(12, 8))
        for xx in X_normalized[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=0.3)
        plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-", linewidth=2)
        plt.title(f"Cluster {yi + 1}")
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Scores')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"cluster_{yi + 1}_combined.png"))
        plt.close()        
        
def davies_bouldin_analysis(X_normalized, n_clusters, output_folder):
    """
    Calculate and save the Davies-Bouldin Index.
    """
    clusterer = TimeSeriesKMeans(n_clusters=n_clusters, random_state=0, metric="dtw")
    cluster_labels = clusterer.fit_predict(X_normalized)
    db_index = davies_bouldin_score(X_normalized.reshape(X_normalized.shape[0], -1), cluster_labels)
    print(f"Davies-Bouldin Index: {db_index}")
    with open(os.path.join(output_folder, "davies_bouldin_combined.txt"), "w") as f:
        f.write(f"Davies-Bouldin Index: {db_index}\n")
    print(f"Davies-Bouldin Index saved to: {os.path.join(output_folder, 'davies_bouldin_combined.txt')}")

combined_data = []
sample_info = [] 

for folder_name in include_folders:
    folder_path = os.path.join(base_dir, folder_name)
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d.lower() not in ignore_folders]
        for file_name in files:
            if file_name == "PCA_scores.txt":
                file_path = os.path.join(root, file_name)
                print(f"Processing file: {file_path}")
                try:
                    data = pd.read_csv(file_path, delimiter='\t', header=None).iloc[row_range[0]:row_range[1], 1:16].to_numpy().T
                    combined_data.append(data)
                    for i in range(data.shape[0]):
                        sample_info.append({"File": os.path.abspath(file_path), 
                                            "Column Index": i + 1})
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if combined_data:
    combined_data = np.vstack(combined_data)
    print(f"Combined data shape: {combined_data.shape}")
    output_folder = os.path.join(base_dir, "DTW_Clustering_Results", "Combined")
    os.makedirs(output_folder, exist_ok=True)
    X_normalized = z_normalization(combined_data)
    plot_normalized_samples(X_normalized, output_folder)
    elbow_method(X_normalized, output_folder)
    best_k = silhouette_analysis(X_normalized)
    cluster_and_visualize(X_normalized, best_k, output_folder, sample_info)
    davies_bouldin_analysis(X_normalized, best_k, output_folder)
else:
    print("No data found in the specified files and folders.")
