import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
# from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram

class Cluster:

    def compute_spearman_distance_columns(self, X):
        # Transpose to shape (n_features, n_samples)
        X_t = X.T
        ranked_X = np.apply_along_axis(rankdata, axis=1, arr=X_t)
        return pairwise_distances(ranked_X, metric='correlation')

    def kmedoids_from_distance_matrix(self, D, k, max_iter=300, random_state=None):
        np.random.seed(random_state)
        m = D.shape[0]
        medoid_indices = np.random.choice(m, k, replace=False)

        for _ in range(max_iter):
            distances_to_medoids = D[:, medoid_indices]
            labels = np.argmin(distances_to_medoids, axis=1)

            new_medoids = []
            for cluster_id in range(k):
                cluster_points = np.where(labels == cluster_id)[0]
                if len(cluster_points) == 0:
                    new_medoids.append(medoid_indices[cluster_id])
                    continue
                submatrix = D[np.ix_(cluster_points, cluster_points)]
                cost = submatrix.sum(axis=1)
                new_medoid = cluster_points[np.argmin(cost)]
                new_medoids.append(new_medoid)

            if np.allclose(medoid_indices, new_medoids):
                break
            medoid_indices = np.array(new_medoids)

        distances_to_medoids = D[:, medoid_indices]
        labels = np.argmin(distances_to_medoids, axis=1)

        return medoid_indices, labels

    def plot_column_radial_dendrogram(self, X, labels, feature_names):
        # Compute distance between ranked columns
        X_t = X.T
        ranked_X = np.apply_along_axis(rankdata, axis=1, arr=X_t)
        dist_matrix = pairwise_distances(ranked_X, metric='correlation')

        # Create linkage matrix
        linkage_matrix = linkage(squareform(dist_matrix), method='average')

        # Generate color palette for clusters
        num_clusters = len(set(labels))
        colors = plt.get_cmap('tab20', num_clusters)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1, polar=True)

        # Get dendrogram data
        dendro = dendrogram(
            linkage_matrix,
            labels=feature_names,
            orientation='left',
            distance_sort='ascending',
            no_plot=True
        )

        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        # Map original feature order to cluster label
        idx_to_label = dict(zip(range(len(feature_names)), labels))
        leaf_order = list(map(int, dendro['leaves']))

        for i, idx in enumerate(leaf_order):
            angle = angles[i]
            cluster_id = idx_to_label[idx]
            ax.plot([angle, angle], [0, 1], color=colors(cluster_id), linewidth=2)
            ax.text(angle, 1.05, feature_names[idx], rotation=np.degrees(angle),
                    ha='right', va='center', fontsize=9, color=colors(cluster_id))

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title('Radial Dendrogram: Feature Clustering (Spearman Distance)', va='bottom')
        plt.tight_layout()
        plt.savefig("E:\Shrimandhar\stock_automation\pythonProject\offline_scripts\ClusterDendro\dendro.png")

if __name__=="__main__":
    obj = Cluster()
    # obj.DataAnalyse()
    df = pd.read_csv("C:\\Users\\Shrimandhar\\Downloads\\stock_data.csv")
    df.drop(columns=["Target"], inplace=True)
    # Step 1: K-Medoids Clustering
    n_clusters = 4
    X = df.values
    feature_names = [i for i in df.columns]

    # Compute distance matrix between columns
    D = obj.compute_spearman_distance_columns(X)

    # Run K-Medoids on columns
    k = 4  # number of feature clusters
    medoid_indices, labels = obj.kmedoids_from_distance_matrix(D, k, random_state=42)



    # Plot dendrogram
    obj.plot_column_radial_dendrogram(X, labels, feature_names)