"""
Standalone Submission Generator - Bypasses Dataset Issues
Reads test.csv directly and creates submission manually
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm


class BaselineModel(nn.Module):
    """Exact copy of your trained model"""
    
    def __init__(self, num_operations=388, num_room_types=11, hidden_dim=512):
        super().__init__()
        
        self.room_encoder = nn.Sequential(
            nn.Linear(num_operations + num_room_types, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )
        
        self.context_encoder = nn.Sequential(
            nn.Linear(num_operations + num_room_types, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_operations)
        
    def forward(self, x, context, context_mask):
        x = x.float()
        context = context.float()
        
        room_features = self.room_encoder(x)
        
        batch_size = context.shape[0]
        context_flat = context.reshape(-1, context.shape[-1])
        context_encoded = self.context_encoder(context_flat)
        context_encoded = context_encoded.reshape(batch_size, -1, context_encoded.shape[-1])
        
        context_mask_expanded = context_mask.unsqueeze(-1).float()
        context_sum = (context_encoded * context_mask_expanded).sum(dim=1)
        context_count = context_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        context_aggregated = context_sum / context_count
        
        combined = torch.cat([room_features, context_aggregated], dim=1)
        features = self.combiner(combined)
        
        logits = self.classifier(features)
        return logits


def cluster_room(room_name):
    """Cluster room names into standard categories"""
    rooms = [
        "andre områder", "kjøkken", "stue", "gang", "soverom",
        "bad", "bod", "vaskerom", "wc", "kjeller", "garasje"
    ]
    
    room_lower = room_name.lower()
    if room_lower in rooms:
        return room_lower
    
    for room in rooms:
        if room in room_lower:
            return room
    
    return "ukjent"


def room_to_onehot(room_cluster):
    """Convert room cluster to one-hot encoding"""
    rooms = [
        "andre områder", "kjøkken", "stue", "gang", "soverom",
        "bad", "bod", "vaskerom", "wc", "kjeller", "garasje"
    ]
    vec = np.zeros(len(rooms), dtype=np.int8)
    if room_cluster in rooms:
        vec[rooms.index(room_cluster)] = 1
    return vec


def ops_to_multihot(op_codes, num_ops=388):
    """Convert list of operation codes to multi-hot vector"""
    vec = np.zeros(num_ops, dtype=np.int8)
    for code in op_codes:
        if 0 <= code < num_ops:
            vec[code] = 1
    return vec


def generate_submission_manual(model_path="best_model.pth", threshold=0.5):
    """Generate submission by manually processing test.csv"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load test data
    print("Loading test.csv...")
    test_df = pd.read_csv("data/test.csv")
    print(f"✓ Loaded {len(test_df)} rows from test.csv")
    
    # Group by project and room
    print("Grouping by project and room...")
    grouped = test_df.groupby(['project_id', 'room']).agg({
        'id': 'first',  # Keep first ID for each room
        'work_operation_cluster_code': list
    }).reset_index()
    
    print(f"✓ Found {len(grouped)} unique rooms across projects")
    
    # Add room clusters
    print("Clustering room names...")
    grouped['room_cluster'] = grouped['room'].apply(cluster_room)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = BaselineModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = {}
    
    with torch.no_grad():
        for project_id in tqdm(grouped['project_id'].unique(), desc="Processing projects"):
            # Get all rooms in this project
            project_rooms = grouped[grouped['project_id'] == project_id]
            
            for idx, room_row in project_rooms.iterrows():
                room_id = room_row['id']
                room_ops = room_row['work_operation_cluster_code']
                room_cluster = room_row['room_cluster']
                
                # Create features for current room
                room_ops_vec = ops_to_multihot(room_ops)
                room_cluster_vec = room_to_onehot(room_cluster)
                room_features = np.concatenate([room_ops_vec, room_cluster_vec])
                
                # Create context from other rooms in project
                other_rooms = project_rooms[project_rooms.index != idx]
                context_features = []
                
                for _, other_room in other_rooms.iterrows():
                    other_ops_vec = ops_to_multihot(other_room['work_operation_cluster_code'])
                    other_cluster_vec = room_to_onehot(other_room['room_cluster'])
                    other_features = np.concatenate([other_ops_vec, other_cluster_vec])
                    context_features.append(other_features)
                
                # Pad context to fixed size (max 50 rooms per project)
                max_context = 50
                if len(context_features) == 0:
                    context_tensor = torch.zeros((1, max_context, 399), dtype=torch.int8)
                    context_mask = torch.zeros((1, max_context), dtype=torch.bool)
                else:
                    context_array = np.array(context_features[:max_context])
                    # Pad if needed
                    if len(context_array) < max_context:
                        padding = np.zeros((max_context - len(context_array), 399), dtype=np.int8)
                        context_array = np.vstack([context_array, padding])
                    
                    context_tensor = torch.tensor(context_array, dtype=torch.int8).unsqueeze(0)
                    context_mask = torch.zeros((1, max_context), dtype=torch.bool)
                    context_mask[0, :len(context_features)] = True
                
                # Prepare input
                x = torch.tensor(room_features, dtype=torch.int8).unsqueeze(0).to(device)
                context_tensor = context_tensor.to(device)
                context_mask = context_mask.to(device)
                
                # Predict
                output = model(x, context_tensor, context_mask)
                probs = torch.sigmoid(output)
                pred = (probs > threshold).cpu().numpy()[0]
                
                # Convert to operation codes (indices where pred == 1)
                pred_codes = [i for i, val in enumerate(pred) if val == 1]
                predictions[room_id] = pred_codes
    
    print(f"\n✓ Generated predictions for {len(predictions)} rooms")
    
    # Statistics
    num_ops = [len(ops) for ops in predictions.values()]
    print(f"\nPrediction statistics:")
    print(f"  Average operations per room: {np.mean(num_ops):.2f}")
    print(f"  Min operations: {min(num_ops)}")
    print(f"  Max operations: {max(num_ops)}")
    print(f"  Rooms with 0 predictions: {sum(1 for x in num_ops if x == 0)}")
    
    # Create submission DataFrame
    print("\nCreating submission file...")
    
    # Get all unique room IDs from test.csv (to ensure we have all 18299)
    all_room_ids = test_df.groupby(['project_id', 'room'])['id'].first().values
    print(f"Total unique room IDs in test.csv: {len(all_room_ids)}")
    
    # Create submission rows
    submission_rows = []
    for room_id in all_room_ids:
        # Get predictions (empty list if not found)
        pred_codes = predictions.get(room_id, [])
        
        # Create multi-hot encoding
        row = [room_id] + ops_to_multihot(pred_codes, 388).tolist()
        submission_rows.append(row)
    
    # Create DataFrame
    columns = ['id'] + [str(i) for i in range(388)]
    submission_df = pd.DataFrame(submission_rows, columns=columns)
    
    print(f"✓ Submission DataFrame shape: {submission_df.shape}")
    
    # Verify
    assert submission_df.shape[0] == 18299, f"Expected 18299 rows, got {submission_df.shape[0]}"
    assert submission_df.shape[1] == 389, f"Expected 389 columns, got {submission_df.shape[1]}"
    assert submission_df['id'].nunique() == 18299, "IDs must be unique"
    
    # Save
    import os
    os.makedirs("submissions", exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submissions/submission_{timestamp}.csv"
    
    submission_df.to_csv(filename, index=False)
    print(f"\n✅ Submission saved to: {filename}")
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"File: {filename}")
    print(f"Rows: {len(submission_df)}")
    print("\nNext steps:")
    print("1. Go to: https://www.kaggle.com/competitions/hackathon-recover-x-cogito/submit")
    print("2. Upload your submission file")
    print("3. Check your score!")
    print("="*60)


if __name__ == "__main__":
    generate_submission_manual()