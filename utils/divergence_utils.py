import numpy as np
from scipy.stats import entropy


class DivergenceMetrics:
 
    @staticmethod
    def renyi_divergence(p, q, alpha=1.0):
        """
        Compute the Rényi Divergence between two distributions.
        """
        p, q = np.array(p), np.array(q)
        if alpha == 1.0:
            return entropy(p, q)
        else:
            return 1 / (alpha - 1) * np.log(np.sum(p**alpha * q ** (1 - alpha)))

 

    @staticmethod
    def wasserstein_distance(p, q):
        """
        Compute the Wasserstein Distance between two distributions.
        """
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        return np.sum(np.abs(cdf_p - cdf_q))

 
    @staticmethod
    def kl_divergence(p, q):
        """
        Compute the Kullback-Leibler (KL) Divergence between two distributions.
        """
        p, q = np.array(p), np.array(q)
        return entropy(p, q)
