import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import silhouette_score



class ModelVisualization:
    def __init__(self, win_pred, lose_pred):
        """
        Initializes the ModelVisualization class with preprocessed win and lose vectors.

        Parameters:
            - win_pred (Tensor): The model's vector representations for 'win' samples (e.g., [B, C]).
            - lose_pred (Tensor): The model's vector representations for 'lose' samples (e.g., [B, C]).
        """
        try:
            # No need to flatten or reduce anymore — input is already [B, C]
            self.win_pred = win_pred[::5].detach().cpu().numpy()   # Shape: [B, C]
            self.lose_pred = lose_pred[::5].detach().cpu().numpy() # Shape: [B, C]
            self.dbs_score = 0
            # self.silhouette = 0
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return
                

    def reduce_dimensionality(self, embeddings, target_dim=3):
        """
        Reduces dimensionality using PCA followed by t-SNE.
        """
        try:
            # Step 1: PCA to 50D
            pca_50d = PCA(n_components=50)
            pca_embeddings = pca_50d.fit_transform(embeddings)

            # Step 2: t-SNE to 3D
            tsne = TSNE(n_components=target_dim, perplexity=30, learning_rate='auto', init='pca', random_state=42)
            reduced_embeddings = tsne.fit_transform(pca_embeddings)
            return reduced_embeddings
        except Exception as e:
            print(f"Dimensionality reduction error: {e}")
            return embeddings

    def plot_3d(self, save_path=None):
        """
        Plots a 3D scatter plot of the model predictions (win and lose) after PCA + t-SNE.
        """
        # Combine win and lose embeddings
        combined_embeddings = np.vstack([self.win_pred, self.lose_pred])
        labels = np.array([0] * len(self.win_pred) + [1] * len(self.lose_pred))

        # Reduce dimensions
        reduced_embeddings = self.reduce_dimensionality(combined_embeddings)

        # Compute Davies–Bouldin score
        self.dbs_score = davies_bouldin_score(reduced_embeddings, labels)
        # self.silhouette = silhouette_score(reduced_embeddings, labels)
        # 3D Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(reduced_embeddings[labels == 0, 0],
                   reduced_embeddings[labels == 0, 1],
                   reduced_embeddings[labels == 0, 2],
                   color='green', label='Not-Hateful', alpha=0.7, marker='o')

        ax.scatter(reduced_embeddings[labels == 1, 0],
                   reduced_embeddings[labels == 1, 1],
                   reduced_embeddings[labels == 1, 2],
                   color='red', label='Hateful', alpha=0.7, marker='x')

        name = save_path.split('.')[0] if save_path else "Model"
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title(f'{name} -- 3D Visualization with PCA + t-SNE -- DBS = {self.dbs_score:.4f}')
        ax.legend(loc='best')

        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
        plt.show()
