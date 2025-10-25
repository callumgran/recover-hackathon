"""
Baseline Solution for Recover Hackathon - FIXED VERSION
This implements a simple neural network baseline using the provided dataloader
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.hackathon import HackathonDataset
from dataset.collate import collate_fn
from metrics import normalized_rooms_score
from tqdm import tqdm
import math
import os
import numpy as np

@torch.no_grad()
def compute_label_counts(train_dataset, batch_size=256):
    """
    Fast pass over the training split to count positive labels per operation.
    Cached to label_counts.npy in the current working directory.
    """
    cache_path = "label_counts.npy"
    if os.path.exists(cache_path):
        arr = np.load(cache_path)
        print(f"[weights] Loaded cached label counts: {arr.shape}")
        return arr

    print("[weights] Computing label counts from training data...")
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    counts = None
    for batch in tqdm(loader, desc="Counting labels"):
        y = batch["Y"]  # [B, num_operations] multi-hot
        if counts is None:
            counts = y.sum(dim=0).cpu().double()  # start in high precision
        else:
            counts += y.sum(dim=0).cpu().double()

    counts = counts.numpy().astype(np.float32)
    np.save(cache_path, counts)
    print(f"[weights] Saved label counts cache to {cache_path}")
    return counts

class BaselineModel(nn.Module):
    """Simple feedforward model with context aggregation"""
    
    def __init__(self, num_operations=388, num_room_types=11, hidden_dim=1024, meta_dim=8, dep_dim=22):
        super().__init__()
        
        # Input: operations (388) + room type (11) + aggregated context
        self.room_encoder = nn.Sequential(
            nn.Linear(num_operations + num_room_types, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )
        
        # Context aggregation (simple mean pooling over context rooms)
        self.context_encoder = nn.Sequential(
            nn.Linear(num_operations + num_room_types, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        # Meta encoder
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),
        )

        # Dependency features encoder
        self.dep_encoder = nn.Sequential(
            nn.Linear(dep_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),
        )

        # Combine room + context + meta
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Output layer
        self.classifier = nn.Linear(hidden_dim // 2, num_operations)
        
    def forward(self, x, context, context_mask, meta, dep):
        x = x.float()
        context = context.float()
        meta = meta.float()
        dep = dep.float()

        room_features = self.room_encoder(x)

        B, C = context.shape[0], context.shape[1]
        context_flat = context.reshape(-1, context.shape[-1])
        context_encoded = self.context_encoder(context_flat)
        context_encoded = context_encoded.reshape(B, C, -1)

        context_mask_expanded = context_mask.unsqueeze(-1).float()
        context_sum = (context_encoded * context_mask_expanded).sum(dim=1)
        context_count = context_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        context_aggregated = context_sum / context_count

        meta_features = self.meta_encoder(meta)
        dep_features = self.dep_encoder(dep)

        combined = torch.cat([room_features, context_aggregated, meta_features, dep_features], dim=1)
        features = self.combiner(combined)
        logits = self.classifier(features)
        return logits

def _get_dep(batch):
    if "deps" in batch:
        return batch["deps"]
    if "dep_feats" in batch:
        return batch["dep_feats"]
    B = batch["meta"].shape[0]
    return torch.zeros((B, 27), dtype=torch.float32, device=batch["meta"].device)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        x = batch["X"].to(device)
        y = batch["Y"].to(device).float()
        context = batch["context"].to(device)
        context_mask = batch["context_mask"].to(device)
        meta = batch["meta"].to(device)
        dep = _get_dep(batch).to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, context, context_mask, meta, dep)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def validate(model, dataloader, device, threshold=0.2):
    model.eval()
    all_preds, all_targets = [], []
    for batch in tqdm(dataloader, desc="Validating"):
        x = batch["X"].to(device)
        y = batch["Y"].to(device)
        context = batch["context"].to(device)
        context_mask = batch["context_mask"].to(device)
        meta = batch["meta"].to(device)
        dep = _get_dep(batch).to(device)

        probs = torch.sigmoid(model(x, context, context_mask, meta, dep))
        preds = (probs > threshold).cpu().numpy()
        targets = y.cpu().numpy()
        for pred, target in zip(preds, targets):
            all_preds.append([i + 1 for i, v in enumerate(pred) if v == 1])
            all_targets.append([i + 1 for i, v in enumerate(target) if v == 1])
    return normalized_rooms_score(all_preds, all_targets)

def main():
    batch_size = 128
    num_epochs = 128
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")

    sampling_strategy = [
		# Full context (no sampling) for strong signal of dependencies
		{"subset_size": 0.7, "sample_pct": 0.70, "use_balanced_data": True,  "use_sampled_calculus": False},
		# Then lighter batches with sampled context to prevent overfitting
		{"subset_size": 0.3, "sample_pct": 0.30, "use_balanced_data": False, "use_sampled_calculus": True},
	]

    train_dataset = HackathonDataset(split="train", download=False, seed=42, sampling_strategy=sampling_strategy)
    val_dataset = HackathonDataset(split="val", download=False, seed=42)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    # Quick sanity: verify meta/dep shapes on first batch
    fb = next(iter(train_loader))
    print("[DEBUG] X:", fb["X"].shape)
    print("[DEBUG] context:", fb["context"].shape)
    print("[DEBUG] meta:", fb["meta"].shape)          # expect [B, 10]
    print("[DEBUG] dep exists:", ("deps" in fb) or ("dep_feats" in fb))
    if ("deps" in fb) or ("dep_feats" in fb):
        dk = "deps" if "deps" in fb else "dep_feats"
        print("[DEBUG] dep shape:", fb[dk].shape)      # expect [B, 27]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = BaselineModel().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_score = float("-inf")
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")

        val_score = validate(model, val_loader, device)
        print(f"Validation Score: {val_score:.4f}")

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_score)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), "best_model_128.pth")
            print("✓ New best score! Model saved.")

        if (epoch + 1) % 3 == 0 and epoch > 0:
            print("Reshuffling training data...")
            train_dataset.shuffle()

    print(f"\n{'='*60}\nTRAINING COMPLETE!\n{'='*60}")
    print(f"Best validation score: {best_score:.4f}")
    print("Model saved to: best_model_128.pth")

    print("\nRunning final validation with best model...")
    model.load_state_dict(torch.load("best_model_128.pth", map_location=device))
    final_score = validate(model, val_loader, device)
    print(f"\nFinal validation score: {final_score:.4f}")

    print("\n" + "="*60)
    print("Next steps:")
    print("1. Run: python3 generate_submission.py")
    print("2. Submit to Kaggle!")
    print("="*60)


if __name__ == "__main__":
    main()