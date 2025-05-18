import numpy as np
import torch


class KernelFunctions:
    def __init__(self, degree=3, c=1, sigma=1.0, weights=None):
        self.degree = degree  # For Polynomial Kernel
        self.c = c  # Constant term for Polynomial Kernel
        self.sigma = sigma  # Bandwidth for RBF Kernel
        self.weights = (
            weights if weights is not None else []
        )  # Weights for Kernel Mixture
        self.eigenvalues = None  # To store eigenvalues for Spectral Kernel
        self.phi_funcs = None  # To store eigenfunctions for Spectral Kernel
        self.cov_matrix = None  # To store the covariance matrix for Mahalanobis Kernel

    def polynomial_kernel(self, u, v):
        """
        Computes the polynomial kernel: K(u, v) = (u ⋅ v + c)^d

        Args:
            u (torch.Tensor): First tensor of shape (batch, dim)
            v (torch.Tensor): Second tensor of shape (batch, dim)

        Returns:
            torch.Tensor: Kernel values.
        """
        # Ensure correct shape for matrix multiplication
        kernel_value = (torch.matmul(u, v.T) + self.c) ** self.degree
        return kernel_value.to(u.device)
        

    def rbf_kernel(self, u, v):
        """
        Computes the RBF kernel: K(u, v) = exp(-||u-v||² / 2σ²)

        Args:
            u (torch.Tensor): First tensor.
            v (torch.Tensor): Second tensor.

        Returns:
            torch.Tensor: RBF kernel value.
        """
        dist = torch.norm(u - v, p=2, dim=1) ** 2  # Squared Euclidean distance
        # print(type(dist))
        kernel_value = torch.exp(-dist / (2 * self.sigma**2))
        return kernel_value.to(u.device)

    def wavelet_kernel(self,u,v):
        dist = torch.norm(u - v, p=2, dim=1) ** 2  # Squared Euclidean distance
        # print(type(dist))
        val=dist / (2 * self.sigma**2)
        exponent_term = torch.exp(-val)
        cosine_term=torch.cos(val)
        wavelet_kernel_value=exponent_term*cosine_term
        return wavelet_kernel_value.to(u.device)


    # def spectral_kernel(self, u, v):
    #     if self.eigenvalues is None or self.phi_funcs is None:
    #         raise ValueError(
    #             "Eigenvalues and eigenfunctions must be computed before using the spectral kernel."
    #         )
    #     diff_sum = 0
    #     for lam, phi in zip(self.eigenvalues, self.phi_funcs):
    #         diff_sum += lam * (phi(u) - phi(v)) ** 2
    #     return diff_sum

    # def mahalanobis_kernel(self, u, v):
    #     if self.cov_matrix is None:
    #         raise ValueError(
    #             "Covariance matrix must be set before using the Mahalanobis kernel."
    #         )
    #     diff = u - v
    #     cov_inv = np.linalg.inv(self.cov_matrix)
    #     return np.exp(-0.5 * np.dot(np.dot(diff.T, cov_inv), diff))

    # Using  x,y to make covariance matrix

    def mahalanobis_kernel(self, u, v):
        # Ensure u and v are column vectors
        u = np.array(u).reshape(-1, 1)
        v = np.array(v).reshape(-1, 1)
        
        # Calculate covariance matrix of u and v
        diff_u = u - np.mean(u)
        diff_v = v - np.mean(v)
        cov_matrix = np.dot(diff_u, diff_v.T) / (u.shape[0] - 1)
        
        # Inverse of the covariance matrix
        cov_inv = np.linalg.inv(cov_matrix)
        
        # Compute Mahalanobis distance
        diff = u - v
        return np.exp(-0.5 * np.dot(np.dot(diff.T, cov_inv), diff))


    # def kernel_mixture(self, u, v, kernels):
    #     if not self.weights or len(self.weights) != len(kernels):
    #         raise ValueError(
    #             "Weights must be provided and match the number of kernels."
    #         )
    #     mixture_sum = 0
    #     for w, kernel in zip(self.weights, kernels):
    #         mixture_sum += w * kernel(u, v)
    #     return mixture_sum

    def compute_kernel(self, u, v, kernel_type):
        if kernel_type == "polynomial":
            return self.polynomial_kernel(u, v)
        elif kernel_type == "rbf":
            return self.rbf_kernel(u, v)
        # elif kernel_type == "spectral":
        #     return self.spectral_kernel(u, v)
        # elif kernel_type == "mahalanobis":
        #     return self.mahalanobis_kernel(u, v)
        else:
            raise ValueError(
                "Unsupported kernel type. Choose from: 'polynomial', 'rbf', 'spectral', 'mahalanobis'."
            )

    # Helper Methods
    @staticmethod
    def compute_covariance_matrix(data):
        """Compute covariance matrix for Mahalanobis kernel."""
        return np.cov(data, rowvar=False)

    @staticmethod
    def compute_eigen_decomposition(cov_matrix):
        """Compute eigenvalues and eigenvectors for Spectral kernel."""
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        return eigenvalues, eigenvectors

    @staticmethod
    def compute_phi_funcs(eigenvectors):
        """Create phi functions (projections) from eigenvectors."""
        phi_funcs = []
        for vec in eigenvectors.T:  # Transpose to get each eigenvector
            phi_funcs.append(lambda x, vec=vec: np.dot(vec, x))
        return phi_funcs


# Example Usage
# data = np.random.rand(100, 5)  # Example dataset
# kf = KernelFunctions(degree=3, c=1, sigma=2.0, weights=[0.4, 0.3, 0.3])

# # Covariance matrix and spectral components
# kf.cov_matrix = kf.compute_covariance_matrix(data)
# eigenvalues, eigenvectors = kf.compute_eigen_decomposition(kf.cov_matrix)
# kf.eigenvalues = eigenvalues
# kf.phi_funcs = kf.compute_phi_funcs(eigenvectors)

# # Define example vectors
# u = np.random.rand(5)
# v = np.random.rand(5)

# # Select and compute a kernel
# kernel_type = "rbf"  # Choose kernel type: Choose from: 'polynomial', 'rbf', 'spectral', 'mahalanobis'
# print(f"{kernel_type.capitalize()} Kernel:", kf.compute_kernel(u, v, kernel_type))
