# 请你帮我写一下python代码，从./datalake途径下面，读取所有的csv的文件，然后构建图中的hierarcical graph (从底到顶，分别是token layer, cell layer, row/column layer, table layer)，包括node_id_mapping(每一个token, cell, row, column, table_name对应的id)，和他们之间的edge_list。包括两块，第一块是不同layer之间的node连接关系，比如一个cell之中对应的所有的token node，都应该和这个cell nod有一条边；一个row/column node之中对应的所有的cell node，都应该和这个row/column node有一条边; 一个table node之中对应的所有的row/column node，都应该和这个table node有一条边。第二块是同一个layer之内的如果有关系的话，应该要有一条边连接起来（这个关系从./label/entity_matching/entity_matching_labels.csv中读取）. 注意，这里哟个的tokenizer是从"/data1/jianweiw/LLM/models_hf/sentence-t5-base"这里获得的

import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict
import json


class HierarchicalGraphBuilder:
    def __init__(self, datalake_path="./datalake", label_path="./label/entity_matching/entity_matching_labels.csv", 
                 training_ratio=0.7, random_seed=42):
        self.datalake_path = datalake_path
        self.label_path = label_path
        self.tokenizer = AutoTokenizer.from_pretrained("/data1/jianweiw/LLM/models_hf/sentence-t5-base")
        self.training_ratio = training_ratio
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Node ID mappings
        self.node_id_mapping = {
            'token': {},     # token -> node_id
            'cell': {},      # (table_name, row_idx, col_idx) -> node_id
            'row': {},       # (table_name, row_idx) -> node_id
            'column': {},    # (table_name, col_idx) -> node_id
            'table': {}      # table_name -> node_id
        }
        
        # Edge lists - each edge is now a dict with source, target, and training_indicator
        self.edge_lists = {
            'token_cell': [],      # token -> cell edges
            'cell_row': [],        # cell -> row edges
            'cell_column': [],     # cell -> column edges
            'row_table': [],       # row -> table edges
            'column_table': [],    # column -> table edges
            'entity_matching': []  # same-layer entity matching edges
        }
        
        # Counter for unique node IDs
        self.node_counter = 0
        
        # Store table data for processing
        self.table_data = {}
    
    def add_edge_with_training_indicator(self, edge_list, source_id, target_id, edge_type="hierarchical"):
        """Add an edge with training indicator"""
        # Determine training indicator based on edge type and random assignment
        if edge_type == "hierarchical":
            # For hierarchical edges, use a deterministic assignment based on node IDs
            training_indicator = int((source_id + target_id) % 100 < self.training_ratio * 100)
        elif edge_type == "entity_matching":
            # For entity matching edges, random assignment
            training_indicator = int(np.random.random() < self.training_ratio)
        else:
            # Default case
            training_indicator = int(np.random.random() < self.training_ratio)
        
    def get_next_node_id(self):
        """Get next unique node ID"""
        self.node_counter += 1
        return self.node_counter
        """Get next unique node ID"""
        self.node_counter += 1
        return self.node_counter
    
    def load_csv_files(self):
        """Load all CSV files from datalake directory"""
        csv_files = []
        for root, dirs, files in os.walk(self.datalake_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                # Use relative path as table name
                table_name = os.path.relpath(csv_file, self.datalake_path)
                table_name = table_name.replace('\\', '/').replace('.csv', '')
                
                df = pd.read_csv(csv_file)
                self.table_data[table_name] = df
                print(f"Loaded table: {table_name}, shape: {df.shape}")
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    def tokenize_cell_content(self, content):
        """Tokenize cell content and return token list"""
        if pd.isna(content):
            content = ""
        else:
            content = str(content)
        
        # Tokenize using the specified tokenizer
        tokens = self.tokenizer.tokenize(content)
        return tokens
    
    def build_table_layer(self):
        """Build table layer nodes"""
        print("Building table layer...")
        for table_name in self.table_data.keys():
            if table_name not in self.node_id_mapping['table']:
                self.node_id_mapping['table'][table_name] = self.get_next_node_id()
    
    def build_row_column_layer(self):
        """Build row and column layer nodes"""
        print("Building row and column layers...")
        for table_name, df in self.table_data.items():
            # Build row nodes
            for row_idx in range(len(df)):
                row_key = (table_name, row_idx)
                if row_key not in self.node_id_mapping['row']:
                    self.node_id_mapping['row'][row_key] = self.get_next_node_id()
            
            # Build column nodes
            for col_idx in range(len(df.columns)):
                col_key = (table_name, col_idx)
                if col_key not in self.node_id_mapping['column']:
                    self.node_id_mapping['column'][col_key] = self.get_next_node_id()
    
    def build_cell_layer(self):
        """Build cell layer nodes"""
        print("Building cell layer...")
        for table_name, df in self.table_data.items():
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    cell_key = (table_name, row_idx, col_idx)
                    if cell_key not in self.node_id_mapping['cell']:
                        self.node_id_mapping['cell'][cell_key] = self.get_next_node_id()
    
    def build_token_layer(self):
        """Build token layer nodes"""
        print("Building token layer...")
        token_counter = 0
        
        for table_name, df in self.table_data.items():
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    cell_content = df.iloc[row_idx, col_idx]
                    tokens = self.tokenize_cell_content(cell_content)
                    
                    for token in tokens:
                        # Create unique token identifier (token + position)
                        token_key = f"{table_name}_{row_idx}_{col_idx}_{token_counter}"
                        if token_key not in self.node_id_mapping['token']:
                            self.node_id_mapping['token'][token_key] = self.get_next_node_id()
                        token_counter += 1
    
    def build_hierarchical_edges(self):
        """Build edges between different layers"""
        print("Building hierarchical edges...")
        
        # Build token -> cell edges
        token_counter = 0
        for table_name, df in self.table_data.items():
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    cell_content = df.iloc[row_idx, col_idx]
                    tokens = self.tokenize_cell_content(cell_content)
                    
                    cell_key = (table_name, row_idx, col_idx)
                    cell_id = self.node_id_mapping['cell'][cell_key]
                    
                    for token in tokens:
                        token_key = f"{table_name}_{row_idx}_{col_idx}_{token_counter}"
                        token_id = self.node_id_mapping['token'][token_key]
                        
                        # Add token -> cell edge with training indicator
                        self.add_edge_with_training_indicator(
                            self.edge_lists['token_cell'], token_id, cell_id, "hierarchical"
                        )
                        token_counter += 1
        
        # Build cell -> row and cell -> column edges
        for table_name, df in self.table_data.items():
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    cell_key = (table_name, row_idx, col_idx)
                    cell_id = self.node_id_mapping['cell'][cell_key]
                    
                    # Cell -> row edge with training indicator
                    row_key = (table_name, row_idx)
                    row_id = self.node_id_mapping['row'][row_key]
                    self.add_edge_with_training_indicator(
                        self.edge_lists['cell_row'], cell_id, row_id, "hierarchical"
                    )
                    
                    # Cell -> column edge with training indicator
                    col_key = (table_name, col_idx)
                    col_id = self.node_id_mapping['column'][col_key]
                    self.add_edge_with_training_indicator(
                        self.edge_lists['cell_column'], cell_id, col_id, "hierarchical"
                    )
        
        # Build row -> table and column -> table edges
        for table_name, df in self.table_data.items():
            table_id = self.node_id_mapping['table'][table_name]
            
            # Row -> table edges with training indicator
            for row_idx in range(len(df)):
                row_key = (table_name, row_idx)
                row_id = self.node_id_mapping['row'][row_key]
                self.add_edge_with_training_indicator(
                    self.edge_lists['row_table'], row_id, table_id, "hierarchical"
                )
            
            # Column -> table edges with training indicator
            for col_idx in range(len(df.columns)):
                col_key = (table_name, col_idx)
                col_id = self.node_id_mapping['column'][col_key]
                self.add_edge_with_training_indicator(
                    self.edge_lists['column_table'], col_id, table_id, "hierarchical"
                )
    
    def build_entity_matching_edges(self):
        """Build edges based on entity matching labels"""
        print("Building entity matching edges...")
        
        try:
            if os.path.exists(self.label_path):
                label_df = pd.read_csv(self.label_path)
                print(f"Loaded entity matching labels: {label_df.shape}")
                
                # Process entity matching labels
                # Assuming the CSV has columns like 'table1', 'row1', 'col1', 'table2', 'row2', 'col2'
                # You may need to adjust this based on your actual label file format
                
                for _, row in label_df.iterrows():
                    try:
                        # Extract matching entity information
                        # Adjust column names based on your actual label file
                        if 'table1' in label_df.columns and 'table2' in label_df.columns:
                            table1 = row['table1']
                            table2 = row['table2']
                            
                            # Add matching edges based on the type of matching
                            if 'row1' in label_df.columns and 'row2' in label_df.columns:
                                row1_key = (table1, row['row1'])
                                row2_key = (table2, row['row2'])
                                
                                if row1_key in self.node_id_mapping['row'] and row2_key in self.node_id_mapping['row']:
                                    id1 = self.node_id_mapping['row'][row1_key]
                                    id2 = self.node_id_mapping['row'][row2_key]
                                    self.add_edge_with_training_indicator(
                                        self.edge_lists['entity_matching'], id1, id2, "entity_matching"
                                    )
                            
                            if 'col1' in label_df.columns and 'col2' in label_df.columns:
                                col1_key = (table1, row['col1'])
                                col2_key = (table2, row['col2'])
                                
                                if col1_key in self.node_id_mapping['column'] and col2_key in self.node_id_mapping['column']:
                                    id1 = self.node_id_mapping['column'][col1_key]
                                    id2 = self.node_id_mapping['column'][col2_key]
                                    self.add_edge_with_training_indicator(
                                        self.edge_lists['entity_matching'], id1, id2, "entity_matching"
                                    )
                    
                    except Exception as e:
                        print(f"Error processing entity matching row: {e}")
                        continue
            
            else:
                print(f"Entity matching label file not found: {self.label_path}")
        
        except Exception as e:
            print(f"Error loading entity matching labels: {e}")
    
    def build_graph(self):
        """Build the complete hierarchical graph"""
        print("Starting to build hierarchical graph...")
        
        # Load CSV files
        self.load_csv_files()
        
        # Build layers from top to bottom
        self.build_table_layer()
        self.build_row_column_layer()
        self.build_cell_layer()
        self.build_token_layer()
        
        # Build edges
        self.build_hierarchical_edges()
        self.build_entity_matching_edges()
        
        print("Graph building completed!")
        self.print_statistics()
    
    def print_statistics(self):
        """Print graph statistics"""
        print("\n=== Graph Statistics ===")
        print(f"Total nodes: {self.node_counter}")
        print(f"Table nodes: {len(self.node_id_mapping['table'])}")
        print(f"Row nodes: {len(self.node_id_mapping['row'])}")
        print(f"Column nodes: {len(self.node_id_mapping['column'])}")
        print(f"Cell nodes: {len(self.node_id_mapping['cell'])}")
        print(f"Token nodes: {len(self.node_id_mapping['token'])}")
        
        print(f"\nEdge counts:")
        print(f"Token -> Cell edges: {len(self.edge_lists['token_cell'])}")
        print(f"Cell -> Row edges: {len(self.edge_lists['cell_row'])}")
        print(f"Cell -> Column edges: {len(self.edge_lists['cell_column'])}")
        print(f"Row -> Table edges: {len(self.edge_lists['row_table'])}")
        print(f"Column -> Table edges: {len(self.edge_lists['column_table'])}")
        print(f"Entity matching edges: {len(self.edge_lists['entity_matching'])}")
    
    def save_graph(self, output_path="./graph_output"):
        """Save the graph to files"""
        os.makedirs(output_path, exist_ok=True)
        
        # Save node mappings
        with open(os.path.join(output_path, 'node_id_mapping.json'), 'w') as f:
            # Convert tuple keys to strings for JSON serialization
            serializable_mapping = {}
            for layer, mapping in self.node_id_mapping.items():
                serializable_mapping[layer] = {}
                for key, value in mapping.items():
                    serializable_mapping[layer][str(key)] = value
            json.dump(serializable_mapping, f, indent=2)
        
        # Save edge lists
        with open(os.path.join(output_path, 'edge_lists.json'), 'w') as f:
            json.dump(self.edge_lists, f, indent=2)
        
        # Save as separate CSV files for each edge type
        for edge_type, edges in self.edge_lists.items():
            if edges:
                # Convert edge dictionaries to DataFrame
                edge_df = pd.DataFrame(edges)
                edge_df.to_csv(os.path.join(output_path, f'{edge_type}_edges.csv'), index=False)
        
        print(f"Graph saved to {output_path}")
    
    def get_node_attributes(self):
        """Get node attributes for analysis"""
        node_attributes = {}
        
        # Add node type attributes
        for layer, mapping in self.node_id_mapping.items():
            for key, node_id in mapping.items():
                node_attributes[node_id] = {
                    'layer': layer,
                    'original_key': str(key)
                }
        
        return node_attributes


# Usage example
if __name__ == "__main__":
    # Initialize the graph builder
    builder = HierarchicalGraphBuilder()
    
    # Build the graph
    builder.build_graph()
    
    # Save the graph
    builder.save_graph()
    
    # Access the results
    print("\nAccessing results:")
    print("Node ID mappings available in: builder.node_id_mapping")
    print("Edge lists available in: builder.edge_lists")
    
    # Example: Get all edges as a single list with training indicators
    all_edges = []
    for edge_type, edges in builder.edge_lists.items():
        for edge in edges:
            all_edges.append((edge['source'], edge['target'], edge['training_indicator'], edge_type))
    
    print(f"Total edges: {len(all_edges)}")
    
    # Example: Count training vs testing edges
    train_edges = sum(1 for edge_type, edges in builder.edge_lists.items() 
                     for edge in edges if edge['training_indicator'] == 1)
    test_edges = sum(1 for edge_type, edges in builder.edge_lists.items() 
                    for edge in edges if edge['training_indicator'] == 0)
    
    print(f"Training edges: {train_edges}")
    print(f"Testing edges: {test_edges}")
    print(f"Training ratio: {train_edges / (train_edges + test_edges):.2f}")