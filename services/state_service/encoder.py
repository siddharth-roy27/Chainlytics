"""
State encoder for converting logistics graph to embeddings
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import networkx as nx
from collections import defaultdict

from .graph_builder import LogisticsGraph, LogisticsGraphNode, LogisticsGraphEdge


class GraphFeatureExtractor:
    """Extract features from logistics graph"""
    
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
    
    def extract_node_features(self, graph: LogisticsGraph) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for each node in the graph
        
        Args:
            graph: LogisticsGraph
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not graph.nodes:
            return np.array([]).reshape(0, 0), []
        
        # Collect all features
        node_features = []
        feature_names = []
        
        # Get all possible node types and attributes
        node_types = set()
        attribute_keys = set()
        
        for node in graph.nodes.values():
            node_types.add(node.node_type)
            attribute_keys.update(node.attributes.keys())
        
        feature_names.extend([f"is_{nt}" for nt in sorted(node_types)])
        feature_names.extend(sorted(attribute_keys))
        feature_names.append("degree")
        feature_names.append("in_degree")
        feature_names.append("out_degree")
        
        # Extract features for each node
        for node_id, node in graph.nodes.items():
            features = []
            
            # One-hot encode node type
            for nt in sorted(node_types):
                features.append(1.0 if node.node_type == nt else 0.0)
            
            # Add attributes
            for attr_key in sorted(attribute_keys):
                attr_value = node.attributes.get(attr_key, 0)
                # Handle different data types
                if isinstance(attr_value, (int, float)):
                    features.append(float(attr_value))
                elif isinstance(attr_value, bool):
                    features.append(1.0 if attr_value else 0.0)
                else:
                    features.append(0.0)  # Default for non-numeric
            
            # Add graph-based features
            if node_id in graph.graph:
                features.append(graph.graph.degree(node_id))
                features.append(graph.graph.in_degree(node_id))
                features.append(graph.graph.out_degree(node_id))
            else:
                features.extend([0, 0, 0])
            
            node_features.append(features)
        
        # Convert to numpy array
        feature_matrix = np.array(node_features)
        
        # Scale features
        if feature_matrix.size > 0:
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
            self.scalers['node_features'] = scaler
        
        return feature_matrix, feature_names
    
    def extract_edge_features(self, graph: LogisticsGraph) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for each edge in the graph
        
        Args:
            graph: LogisticsGraph
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not graph.edges:
            return np.array([]).reshape(0, 0), []
        
        # Collect all possible edge attributes
        attribute_keys = set()
        for edge in graph.edges:
            attribute_keys.update(edge.attributes.keys())
        
        feature_names = sorted(attribute_keys)
        feature_names.extend(["is_transport_route", "is_supply_link", "is_distribution_link"])
        
        # Extract features for each edge
        edge_features = []
        for edge in graph.edges:
            features = []
            
            # Add attributes
            for attr_key in sorted(attribute_keys):
                attr_value = edge.attributes.get(attr_key, 0)
                if isinstance(attr_value, (int, float)):
                    features.append(float(attr_value))
                elif isinstance(attr_value, bool):
                    features.append(1.0 if attr_value else 0.0)
                else:
                    features.append(0.0)
            
            # One-hot encode edge type
            edge_type_features = [
                1.0 if edge.edge_type == "transport_route" else 0.0,
                1.0 if edge.edge_type == "supply_link" else 0.0,
                1.0 if edge.edge_type == "distribution_link" else 0.0
            ]
            features.extend(edge_type_features)
            
            edge_features.append(features)
        
        # Convert to numpy array
        feature_matrix = np.array(edge_features)
        
        # Scale features
        if feature_matrix.size > 0:
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
            self.scalers['edge_features'] = scaler
        
        return feature_matrix, feature_names
    
    def extract_graph_level_features(self, graph: LogisticsGraph) -> Dict[str, float]:
        """
        Extract graph-level features
        
        Args:
            graph: LogisticsGraph
            
        Returns:
            Dictionary of graph-level features
        """
        features = {}
        
        # Basic statistics
        features['num_nodes'] = len(graph.nodes)
        features['num_edges'] = len(graph.edges)
        
        if len(graph.nodes) > 0:
            features['density'] = len(graph.edges) / (len(graph.nodes) * (len(graph.nodes) - 1))
            features['avg_degree'] = len(graph.edges) * 2 / len(graph.nodes)
        else:
            features['density'] = 0.0
            features['avg_degree'] = 0.0
        
        # Connected components
        if len(graph.graph.nodes()) > 0:
            try:
                features['num_connected_components'] = nx.number_weakly_connected_components(graph.graph)
                features['largest_component_ratio'] = (
                    len(max(nx.weakly_connected_components(graph.graph), key=len)) / 
                    len(graph.graph.nodes())
                )
            except:
                features['num_connected_components'] = 1
                features['largest_component_ratio'] = 1.0
        else:
            features['num_connected_components'] = 0
            features['largest_component_ratio'] = 0.0
        
        # Centrality measures (sample a few nodes for efficiency)
        if len(graph.graph.nodes()) > 0:
            try:
                # Degree centrality
                degree_centrality = nx.degree_centrality(graph.graph)
                features['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
                features['max_degree_centrality'] = np.max(list(degree_centrality.values()))
                
                # Betweenness centrality (sample for large graphs)
                nodes_sample = list(graph.graph.nodes())[:100]  # Limit for performance
                betweenness = nx.betweenness_centrality(graph.graph, k=len(nodes_sample))
                features['avg_betweenness_centrality'] = np.mean(list(betweenness.values()))
                features['max_betweenness_centrality'] = np.max(list(betweenness.values()))
            except:
                features['avg_degree_centrality'] = 0.0
                features['max_degree_centrality'] = 0.0
                features['avg_betweenness_centrality'] = 0.0
                features['max_betweenness_centrality'] = 0.0
        
        return features


class StateEncoder:
    """Encode logistics state as embeddings"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.feature_extractor = GraphFeatureExtractor()
        self.node_encoder = None
        self.edge_encoder = None
    
    def encode_graph_nodes(self, graph: LogisticsGraph) -> np.ndarray:
        """
        Encode graph nodes as embeddings
        
        Args:
            graph: LogisticsGraph
            
        Returns:
            Node embeddings matrix (num_nodes, embedding_dim)
        """
        # Extract node features
        node_features, _ = self.feature_extractor.extract_node_features(graph)
        
        if node_features.size == 0:
            return np.array([]).reshape(0, self.embedding_dim)
        
        # If we have fewer features than embedding dimension, pad with zeros
        if node_features.shape[1] < self.embedding_dim:
            padding = np.zeros((node_features.shape[0], self.embedding_dim - node_features.shape[1]))
            embeddings = np.hstack([node_features, padding])
        # If we have more features, use PCA-like projection (simple truncation for now)
        elif node_features.shape[1] > self.embedding_dim:
            embeddings = node_features[:, :self.embedding_dim]
        else:
            embeddings = node_features
        
        return embeddings.astype(np.float32)
    
    def encode_graph_structure(self, graph: LogisticsGraph) -> np.ndarray:
        """
        Encode graph structure as adjacency matrix embedding
        
        Args:
            graph: LogisticsGraph
            
        Returns:
            Adjacency matrix embedding (num_nodes, num_nodes)
        """
        if len(graph.nodes) == 0:
            return np.array([]).reshape(0, 0)
        
        # Create adjacency matrix
        node_ids = list(graph.nodes.keys())
        n_nodes = len(node_ids)
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # Fill adjacency matrix
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        for edge in graph.edges:
            if edge.source_id in node_id_to_idx and edge.target_id in node_id_to_idx:
                src_idx = node_id_to_idx[edge.source_id]
                tgt_idx = node_id_to_idx[edge.target_id]
                # Use edge weight if available, otherwise 1
                weight = edge.attributes.get('weight', 1.0)
                adj_matrix[src_idx, tgt_idx] = weight
        
        return adj_matrix.astype(np.float32)
    
    def encode_complete_state(self, graph: LogisticsGraph) -> Dict[str, np.ndarray]:
        """
        Encode complete state including nodes, edges, and graph-level features
        
        Args:
            graph: LogisticsGraph
            
        Returns:
            Dictionary with different state encodings
        """
        encodings = {}
        
        # Node embeddings
        encodings['node_embeddings'] = self.encode_graph_nodes(graph)
        
        # Structure embedding
        encodings['adjacency_matrix'] = self.encode_graph_structure(graph)
        
        # Graph-level features
        graph_features = self.feature_extractor.extract_graph_level_features(graph)
        encodings['graph_features'] = np.array(list(graph_features.values()), dtype=np.float32)
        
        # Edge features
        edge_features, _ = self.feature_extractor.extract_edge_features(graph)
        encodings['edge_features'] = edge_features
        
        return encodings
    
    def get_state_dimension(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get dimensions of state encodings
        
        Returns:
            Dictionary mapping encoding names to their dimensions
        """
        # Note: Actual dimensions depend on the graph, so we return placeholders
        return {
            'node_embeddings': ('num_nodes', self.embedding_dim),
            'adjacency_matrix': ('num_nodes', 'num_nodes'),
            'graph_features': ('num_graph_features',),  # ~10-15 features
            'edge_features': ('num_edges', 'num_edge_features')  # Variable
        }


# Example usage
if __name__ == "__main__":
    # Create sample graph
    from .graph_builder import LogisticsGraphBuilder
    
    # Sample data
    warehouses_data = {
        'warehouse_id': ['WH001', 'WH002', 'WH003'],
        'latitude': [40.7128, 34.0522, 41.8781],
        'longitude': [-74.0060, -118.2437, -87.6298],
        'capacity': [10000, 15000, 12000],
        'current_inventory': [8000, 11000, 9500],
        'demand_forecast': [9000, 13000, 10000]
    }
    warehouses_df = pd.DataFrame(warehouses_data)
    
    routes_data = {
        'origin_warehouse_id': ['WH001', 'WH001', 'WH002'],
        'destination_warehouse_id': ['WH002', 'WH003', 'WH003'],
        'distance': [250, 300, 180],
        'travel_time': [5, 6, 3],
        'cost': [500, 600, 360]
    }
    routes_df = pd.DataFrame(routes_data)
    
    # Build graph
    builder = LogisticsGraphBuilder()
    graph = builder.build_complete_graph(warehouses_df, routes_df)
    
    # Encode state
    encoder = StateEncoder(embedding_dim=64)
    state_encoding = encoder.encode_complete_state(graph)
    
    print("State Encoding Results:")
    print(f"Node embeddings shape: {state_encoding['node_embeddings'].shape}")
    print(f"Adjacency matrix shape: {state_encoding['adjacency_matrix'].shape}")
    print(f"Graph features shape: {state_encoding['graph_features'].shape}")
    print(f"Edge features shape: {state_encoding['edge_features'].shape}")