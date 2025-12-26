"""
Graph builder for logistics state representation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx


@dataclass
class LogisticsGraphNode:
    """Node in the logistics graph"""
    node_id: str
    node_type: str  # 'warehouse', 'distribution_center', 'retail_store', 'supplier', 'customer'
    attributes: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[float, float] = None  # (latitude, longitude)


@dataclass
class LogisticsGraphEdge:
    """Edge in the logistics graph"""
    source_id: str
    target_id: str
    edge_type: str  # 'transport_route', 'supply_link', 'distribution_link'
    attributes: Dict[str, Any] = field(default_factory=dict)


class LogisticsGraph:
    """Logistics graph representation"""
    
    def __init__(self):
        self.nodes: Dict[str, LogisticsGraphNode] = {}
        self.edges: List[LogisticsGraphEdge] = []
        self.graph = nx.DiGraph()  # NetworkX graph for algorithms
    
    def add_node(self, node: LogisticsGraphNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, 
                           node_type=node.node_type, 
                           attributes=node.attributes,
                           position=node.position)
    
    def add_edge(self, edge: LogisticsGraphEdge) -> None:
        """Add an edge to the graph"""
        self.edges.append(edge)
        self.graph.add_edge(edge.source_id, edge.target_id,
                           edge_type=edge.edge_type,
                           attributes=edge.attributes)
    
    def get_node(self, node_id: str) -> Optional[LogisticsGraphNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node"""
        if node_id in self.graph:
            return list(self.graph.neighbors(node_id))
        return []
    
    def get_predecessors(self, node_id: str) -> List[str]:
        """Get predecessors of a node"""
        if node_id in self.graph:
            return list(self.graph.predecessors(node_id))
        return []
    
    def get_subgraph(self, node_ids: List[str]) -> 'LogisticsGraph':
        """Get subgraph containing specified nodes"""
        subgraph = LogisticsGraph()
        
        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
        
        # Add edges between selected nodes
        for edge in self.edges:
            if edge.source_id in node_ids and edge.target_id in node_ids:
                subgraph.add_edge(edge)
        
        return subgraph


class LogisticsGraphBuilder:
    """Build logistics graphs from raw data"""
    
    def __init__(self):
        self.node_counter = defaultdict(int)
    
    def build_from_warehouses(self, warehouses_df: pd.DataFrame) -> LogisticsGraph:
        """
        Build graph from warehouse data
        
        Args:
            warehouses_df: DataFrame with warehouse data
                Expected columns: warehouse_id, latitude, longitude, capacity, 
                current_inventory, demand_forecast, type
            
        Returns:
            LogisticsGraph
        """
        graph = LogisticsGraph()
        
        # Add warehouse nodes
        for _, row in warehouses_df.iterrows():
            node = LogisticsGraphNode(
                node_id=row['warehouse_id'],
                node_type='warehouse',
                attributes={
                    'capacity': row.get('capacity', 0),
                    'current_inventory': row.get('current_inventory', 0),
                    'demand_forecast': row.get('demand_forecast', 0),
                    'type': row.get('type', 'standard')
                },
                position=(row.get('latitude', 0.0), row.get('longitude', 0.0))
            )
            graph.add_node(node)
        
        return graph
    
    def build_from_routes(self, routes_df: pd.DataFrame, 
                         existing_graph: LogisticsGraph = None) -> LogisticsGraph:
        """
        Add route edges to existing graph or create new graph
        
        Args:
            routes_df: DataFrame with route data
                Expected columns: origin_warehouse_id, destination_warehouse_id,
                distance, travel_time, cost, capacity_utilization
            existing_graph: Existing graph to add edges to (optional)
            
        Returns:
            LogisticsGraph
        """
        if existing_graph is None:
            graph = LogisticsGraph()
        else:
            graph = existing_graph
        
        # Add route edges
        for _, row in routes_df.iterrows():
            edge = LogisticsGraphEdge(
                source_id=row['origin_warehouse_id'],
                target_id=row['destination_warehouse_id'],
                edge_type='transport_route',
                attributes={
                    'distance': row.get('distance', 0),
                    'travel_time': row.get('travel_time', 0),
                    'cost': row.get('cost', 0),
                    'capacity_utilization': row.get('capacity_utilization', 0)
                }
            )
            graph.add_edge(edge)
        
        return graph
    
    def build_complete_graph(self, warehouses_df: pd.DataFrame,
                           routes_df: pd.DataFrame) -> LogisticsGraph:
        """
        Build complete logistics graph from warehouses and routes
        
        Args:
            warehouses_df: DataFrame with warehouse data
            routes_df: DataFrame with route data
            
        Returns:
            LogisticsGraph
        """
        # Build from warehouses
        graph = self.build_from_warehouses(warehouses_df)
        
        # Add routes
        graph = self.build_from_routes(routes_df, graph)
        
        return graph
    
    def build_dynamic_graph(self, current_state: Dict) -> LogisticsGraph:
        """
        Build graph from current system state
        
        Args:
            current_state: Dictionary with current state information
            
        Returns:
            LogisticsGraph
        """
        graph = LogisticsGraph()
        
        # Add warehouse nodes
        warehouses = current_state.get('warehouses', [])
        for warehouse in warehouses:
            node = LogisticsGraphNode(
                node_id=warehouse['id'],
                node_type='warehouse',
                attributes={
                    'capacity': warehouse.get('capacity', 0),
                    'current_inventory': warehouse.get('inventory', 0),
                    'demand_forecast': warehouse.get('forecast', 0),
                    'status': warehouse.get('status', 'active')
                },
                position=(warehouse.get('lat', 0.0), warehouse.get('lon', 0.0))
            )
            graph.add_node(node)
        
        # Add supplier nodes
        suppliers = current_state.get('suppliers', [])
        for supplier in suppliers:
            node = LogisticsGraphNode(
                node_id=supplier['id'],
                node_type='supplier',
                attributes={
                    'reliability': supplier.get('reliability', 1.0),
                    'lead_time': supplier.get('lead_time', 0),
                    'capacity': supplier.get('capacity', 0)
                },
                position=(supplier.get('lat', 0.0), supplier.get('lon', 0.0))
            )
            graph.add_node(node)
        
        # Add customer nodes
        customers = current_state.get('customers', [])
        for customer in customers:
            node = LogisticsGraphNode(
                node_id=customer['id'],
                node_type='customer',
                attributes={
                    'demand': customer.get('demand', 0),
                    'priority': customer.get('priority', 1),
                    'service_level': customer.get('service_level', 0.95)
                },
                position=(customer.get('lat', 0.0), customer.get('lon', 0.0))
            )
            graph.add_node(node)
        
        # Add edges
        connections = current_state.get('connections', [])
        for conn in connections:
            edge = LogisticsGraphEdge(
                source_id=conn['source'],
                target_id=conn['target'],
                edge_type=conn.get('type', 'unknown'),
                attributes={
                    'cost': conn.get('cost', 0),
                    'time': conn.get('time', 0),
                    'capacity': conn.get('capacity', 0),
                    'utilization': conn.get('utilization', 0)
                }
            )
            graph.add_edge(edge)
        
        return graph
    
    def get_graph_statistics(self, graph: LogisticsGraph) -> Dict[str, Any]:
        """Get statistics about the graph"""
        stats = {
            'node_count': len(graph.nodes),
            'edge_count': len(graph.edges),
            'node_types': defaultdict(int),
            'edge_types': defaultdict(int),
            'isolated_nodes': 0,
            'density': 0.0
        }
        
        # Count node types
        for node in graph.nodes.values():
            stats['node_types'][node.node_type] += 1
        
        # Count edge types
        for edge in graph.edges:
            stats['edge_types'][edge.edge_type] += 1
        
        # Count isolated nodes
        isolated = 0
        for node_id in graph.graph.nodes():
            if graph.graph.degree(node_id) == 0:
                isolated += 1
        stats['isolated_nodes'] = isolated
        
        # Calculate density
        n = len(graph.nodes)
        if n > 1:
            max_edges = n * (n - 1)  # Directed graph
            stats['density'] = len(graph.edges) / max_edges if max_edges > 0 else 0.0
        
        return dict(stats)


# Example usage
if __name__ == "__main__":
    # Create sample data
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
    
    # Print statistics
    stats = builder.get_graph_statistics(graph)
    print("Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")