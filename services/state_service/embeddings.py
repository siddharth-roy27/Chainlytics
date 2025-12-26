"""
Embeddings management for logistics state representation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
from datetime import datetime


class GraphEmbeddingModel(nn.Module):
    """Neural network for learning graph embeddings"""
    
    def __init__(self, node_feature_dim: int, embedding_dim: int = 128, 
                 hidden_dims: List[int] = None):
        """
        Initialize the graph embedding model
        
        Args:
            node_feature_dim: Dimension of input node features
            embedding_dim: Dimension of output embeddings
            hidden_dims: List of hidden layer dimensions
        """
        super(GraphEmbeddingModel, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Build layers
        layers = []
        input_dim = node_feature_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, embedding_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Additional layers for graph convolution (simplified)
        self.graph_conv = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, node_features: torch.Tensor, 
                adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute embeddings
        
        Args:
            node_features: Node features tensor (batch_size, num_nodes, feature_dim)
            adjacency_matrix: Adjacency matrix (batch_size, num_nodes, num_nodes)
            
        Returns:
            Node embeddings (batch_size, num_nodes, embedding_dim)
        """
        # Process node features through MLP
        embeddings = self.mlp(node_features)
        
        # Simple graph convolution (mean aggregation)
        # In practice, you might want to use more sophisticated GNN layers
        normalized_adj = F.normalize(adjacency_matrix, p=1, dim=-1)
        graph_embeddings = torch.bmm(normalized_adj, embeddings)
        
        # Combine with original embeddings
        combined_embeddings = embeddings + self.graph_conv(graph_embeddings)
        
        # Normalize embeddings
        normalized_embeddings = F.normalize(combined_embeddings, p=2, dim=-1)
        
        return normalized_embeddings


class EmbeddingManager:
    """Manage state embeddings and their persistence"""
    
    def __init__(self, embedding_dim: int = 128, model_path: str = None):
        """
        Initialize embedding manager
        
        Args:
            embedding_dim: Dimension of embeddings
            model_path: Path to pre-trained model (optional)
        """
        self.embedding_dim = embedding_dim
        self.model = None
        self.model_path = model_path
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def initialize_model(self, node_feature_dim: int, 
                        hidden_dims: List[int] = None) -> None:
        """
        Initialize the embedding model
        
        Args:
            node_feature_dim: Dimension of input node features
            hidden_dims: Hidden layer dimensions
        """
        self.model = GraphEmbeddingModel(
            node_feature_dim=node_feature_dim,
            embedding_dim=self.embedding_dim,
            hidden_dims=hidden_dims
        )
    
    def compute_embeddings(self, node_features: np.ndarray, 
                         adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Compute embeddings for graph state
        
        Args:
            node_features: Node features array (num_nodes, feature_dim)
            adjacency_matrix: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Embeddings array (num_nodes, embedding_dim)
        """
        if self.model is None:
            # If no model, return simple feature-based embeddings
            return self._compute_simple_embeddings(node_features)
        
        # Convert to tensors
        node_features_tensor = torch.FloatTensor(node_features).unsqueeze(0)  # Add batch dimension
        adj_matrix_tensor = torch.FloatTensor(adjacency_matrix).unsqueeze(0)
        
        # Compute embeddings
        with torch.no_grad():
            embeddings = self.model(node_features_tensor, adj_matrix_tensor)
        
        # Convert back to numpy
        return embeddings.squeeze(0).numpy()
    
    def _compute_simple_embeddings(self, node_features: np.ndarray) -> np.ndarray:
        """
        Compute simple embeddings when no model is available
        
        Args:
            node_features: Node features array
            
        Returns:
            Simple embeddings
        """
        # Simple approach: normalize and pad/truncate features
        if node_features.size == 0:
            return np.zeros((0, self.embedding_dim))
        
        # Normalize features
        normalized_features = (node_features - np.mean(node_features, axis=0)) / (
            np.std(node_features, axis=0) + 1e-8
        )
        
        # Pad or truncate to embedding dimension
        if normalized_features.shape[1] < self.embedding_dim:
            # Pad with zeros
            padding = np.zeros((
                normalized_features.shape[0], 
                self.embedding_dim - normalized_features.shape[1]
            ))
            embeddings = np.hstack([normalized_features, padding])
        elif normalized_features.shape[1] > self.embedding_dim:
            # Truncate
            embeddings = normalized_features[:, :self.embedding_dim]
        else:
            embeddings = normalized_features
        
        # Normalize embeddings
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm[norm == 0] = 1
        embeddings = embeddings / norm
        
        return embeddings
    
    def save_model(self, path: str) -> None:
        """
        Save the embedding model
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self.model_path = path
    
    def load_model(self, path: str) -> None:
        """
        Load a pre-trained embedding model
        
        Args:
            path: Path to model file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # We need to know the input dimensions to initialize the model
        # This is a simplified approach - in practice, you might save this information
        checkpoint = torch.load(path)
        
        # Extract input dimension from checkpoint (this is a simplification)
        # In practice, you'd save this information separately
        if hasattr(checkpoint, 'mlp.0.weight'):
            node_feature_dim = checkpoint['mlp.0.weight'].shape[1]
        else:
            # Default assumption
            node_feature_dim = 64
        
        # Initialize model
        self.initialize_model(node_feature_dim)
        
        # Load weights
        self.model.load_state_dict(checkpoint)
        self.model_path = path
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       metadata: Dict[str, Any], 
                       path: str) -> None:
        """
        Save computed embeddings with metadata
        
        Args:
            embeddings: Embeddings array
            metadata: Metadata dictionary
            path: Path to save embeddings
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'embeddings': embeddings,
            'metadata': metadata,
            'saved_at': datetime.now().isoformat(),
            'embedding_dim': self.embedding_dim
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_embeddings(self, path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load saved embeddings
        
        Args:
            path: Path to embeddings file
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        return data['embeddings'], data['metadata']


class EmbeddingCache:
    """Cache for storing and retrieving embeddings"""
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize embedding cache
        
        Args:
            cache_size: Maximum number of embeddings to cache
        """
        self.cache_size = cache_size
        self.cache = {}  # state_hash -> (embeddings, timestamp)
        self.access_order = []  # LRU tracking
    
    def get(self, state_hash: str) -> Optional[np.ndarray]:
        """
        Get embeddings from cache
        
        Args:
            state_hash: Hash of state to retrieve
            
        Returns:
            Embeddings if found, None otherwise
        """
        if state_hash in self.cache:
            # Update access order for LRU
            if state_hash in self.access_order:
                self.access_order.remove(state_hash)
            self.access_order.append(state_hash)
            
            return self.cache[state_hash][0]
        return None
    
    def put(self, state_hash: str, embeddings: np.ndarray) -> None:
        """
        Put embeddings in cache
        
        Args:
            state_hash: Hash of state
            embeddings: Embeddings to cache
        """
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        # Add new entry
        self.cache[state_hash] = (embeddings, datetime.now())
        self.access_order.append(state_hash)
    
    def invalidate(self, state_hash: str) -> None:
        """
        Remove entry from cache
        
        Args:
            state_hash: Hash of state to remove
        """
        if state_hash in self.cache:
            del self.cache[state_hash]
        if state_hash in self.access_order:
            self.access_order.remove(state_hash)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    num_nodes = 5
    feature_dim = 10
    
    node_features = np.random.randn(num_nodes, feature_dim)
    adjacency_matrix = np.random.rand(num_nodes, num_nodes)
    
    # Create embedding manager
    embedding_manager = EmbeddingManager(embedding_dim=64)
    
    # Compute embeddings
    embeddings = embedding_manager.compute_embeddings(node_features, adjacency_matrix)
    
    print(f"Computed embeddings shape: {embeddings.shape}")
    print(f"Embedding norms: {np.linalg.norm(embeddings, axis=1)}")
    
    # Test caching
    cache = EmbeddingCache(cache_size=10)
    state_hash = "test_state_001"
    
    cache.put(state_hash, embeddings)
    cached_embeddings = cache.get(state_hash)
    
    if cached_embeddings is not None:
        print("Successfully retrieved embeddings from cache")
        print(f"Cached embeddings shape: {cached_embeddings.shape}")
</file>