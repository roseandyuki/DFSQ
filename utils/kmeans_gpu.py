import torch
import numpy as np
from typing import Optional, Tuple


class KMeansGPU:
    """
    GPU-accelerated K-means clustering using PyTorch
    
    Args:
        n_clusters: Number of clusters
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        n_init: Number of times to run with different centroid seeds
        device: PyTorch device ('cuda' or 'cpu')
        verbose: Whether to print progress
    """
    
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 1000,
        tol: float = 1e-4,
        n_init: int = 10,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.centers = None  # Alias for compatibility with existing code
    
    def fit(self, X: np.ndarray) -> 'KMeansGPU':
        """
        Fit K-means clustering
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            self
        """
        # Convert to torch tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float().to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        n_samples, n_features = X_tensor.shape
        
        best_inertia = None
        best_centers = None
        best_labels = None
        
        for init_idx in range(self.n_init):
            # Initialize centers using k-means++
            centers = self._init_centers_kmeans_plus_plus(X_tensor, self.n_clusters)
            
            for iteration in range(self.max_iter):
                # Assign samples to nearest center
                labels = self._assign_labels(X_tensor, centers)
                
                # Update centers
                new_centers = self._update_centers(X_tensor, labels, self.n_clusters)
                
                # Check convergence
                center_shift = torch.sum(torch.sqrt(torch.sum((centers - new_centers) ** 2, dim=1)))
                
                centers = new_centers
                
                if center_shift < self.tol:
                    if self.verbose:
                        print(f"  Init {init_idx + 1}/{self.n_init}: Converged at iteration {iteration + 1}")
                    break
            
            # Compute inertia (sum of squared distances to centers)
            inertia = self._compute_inertia(X_tensor, labels, centers)
            
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
        
        self.cluster_centers_ = best_centers.cpu().numpy()
        self.labels_ = best_labels.cpu().numpy()
        self.inertia_ = best_inertia.item()
        self.centers = self.cluster_centers_  # Alias for compatibility
        
        return self
    
    def _init_centers_kmeans_plus_plus(self, X: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """
        Initialize centers using k-means++ algorithm
        """
        n_samples = X.shape[0]
        
        # Choose first center randomly
        center_idx = torch.randint(0, n_samples, (1,), device=self.device)
        centers = X[center_idx]
        
        for _ in range(1, n_clusters):
            # Compute distances to nearest center
            distances = torch.cdist(X, centers).min(dim=1)[0]
            
            # Choose next center with probability proportional to distance squared
            probabilities = distances ** 2
            probabilities = probabilities / probabilities.sum()
            
            center_idx = torch.multinomial(probabilities, 1)
            centers = torch.cat([centers, X[center_idx]], dim=0)
        
        return centers
    
    def _assign_labels(self, X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        Assign each sample to nearest center
        """
        # Compute pairwise distances efficiently
        distances = torch.cdist(X, centers)
        labels = torch.argmin(distances, dim=1)
        return labels
    
    def _update_centers(self, X: torch.Tensor, labels: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """
        Update centers as mean of assigned samples
        """
        centers = torch.zeros(n_clusters, X.shape[1], device=self.device, dtype=X.dtype)
        
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = X[mask].mean(dim=0)
            else:
                # If no samples assigned, reinitialize randomly
                centers[k] = X[torch.randint(0, X.shape[0], (1,), device=self.device)]
        
        return centers
    
    def _compute_inertia(self, X: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        Compute sum of squared distances to centers
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                inertia += torch.sum((X[mask] - centers[k]) ** 2)
        return inertia
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet")
        
        X_tensor = torch.from_numpy(X).float().to(self.device)
        centers_tensor = torch.from_numpy(self.cluster_centers_).float().to(self.device)
        
        labels = self._assign_labels(X_tensor, centers_tensor)
        return labels.cpu().numpy()
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels (for compatibility with existing code)
        """
        return self.predict(X)
