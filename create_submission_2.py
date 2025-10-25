# submit_from_loader.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset.hackathon import HackathonDataset
from dataset.collate import collate_fn_infer, collate_fn

from metrics import normalized_rooms_score

@torch.no_grad()
def validate_loader(model, val_loader, device, threshold=0.2):
    model.eval()
    all_preds, all_targets = [], []
    for batch in tqdm(val_loader, desc="Validating"):
        x = batch["X"].to(device, dtype=torch.float32)
        y = batch["Y"].to(device)  # multi-hot labels
        context = batch["context"].to(device, dtype=torch.float32)
        context_mask = batch["context_mask"].to(device)
        meta = batch["meta"].to(device, dtype=torch.float32)
        dep = batch["deps"].to(device, dtype=torch.float32)

        logits = model(x, context, context_mask, meta, dep)
        probs = torch.sigmoid(logits).cpu()

        preds_bin = (probs > threshold).to(torch.int)
        targets_bin = y.cpu().to(torch.int)

        for i in range(preds_bin.shape[0]):
            pred_codes = [j + 1 for j in torch.nonzero(preds_bin[i], as_tuple=False).view(-1).tolist()]
            tgt_codes  = [j + 1 for j in torch.nonzero(targets_bin[i], as_tuple=False).view(-1).tolist()]
            all_preds.append(pred_codes)
            all_targets.append(tgt_codes)

    score = normalized_rooms_score(all_preds, all_targets)
    return score

def sweep_threshold(model, val_loader, device, grid=None):
    if grid is None:
        # grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
        grid = [0.17]
    scores = []
    for t in grid:
        s = validate_loader(model, val_loader, device, threshold=t)
        print(f"threshold={t:.2f} -> val score={s:.4f}")
        scores.append((s, t))
    best_score, best_t = max(scores, key=lambda x: x[0])
    print(f"\nBest threshold on val: {best_t:.2f} (score={best_score:.4f})")
    return best_t, best_score

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

def main(
    model_path="best_model_128.pth",
    data_root="data",
    batch_size=128,
    threshold=0.2,
    do_validate=True,
):
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---------- (optional) validation ----------
    if do_validate:
        val_dataset = HackathonDataset(split="val", download=False, seed=42, root=data_root)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,      # <-- uses labels + prebuilt meta
            pin_memory=torch.cuda.is_available(),
        )
    # ---------- load model ----------
    model = BaselineModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    best_t = threshold
    if do_validate:
        best_t, best_score = sweep_threshold(model, val_loader, device)
        print(f"\nValidation score @ threshold={best_t:.2f}: {best_score:.4f}\n")

    # ---------- submit on test ----------
    test_df = pd.read_csv(os.path.join(data_root, "test.csv"))
    id_by_key = test_df.groupby(["project_id", "room"])["id"].first().to_dict()
    all_ids = test_df.groupby(["project_id", "room"])["id"].first().values

    test_dataset = HackathonDataset(split="test", download=False, seed=42, root=data_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_infer,   # <-- test-time infer collate
        pin_memory=torch.cuda.is_available(),
    )

    predictions: dict[int, list[int]] = {}
    unresolved = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            x = batch["X"].to(device, dtype=torch.float32)
            context = batch["context"].to(device, dtype=torch.float32)
            context_mask = batch["context_mask"].to(device)
            meta = batch["meta"].to(device, dtype=torch.float32)
            dep = batch["deps"].to(device, dtype=torch.float32)

            ids_tensor = batch["ids"]         # -1 if missing
            proj_tensor = batch["project_id"]
            rooms_list = batch["room"]

            logits = model(x, context, context_mask, meta, dep)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(probs.shape[0]):
                row = probs[i]
                pred_codes = np.flatnonzero(row > best_t).astype(int).tolist()

                rid = int(ids_tensor[i].item())
                if rid == -1:
                    key = (int(proj_tensor[i].item()), rooms_list[i])
                    rid = id_by_key.get(key, None)
                if rid is None:
                    unresolved += 1
                    continue
                predictions[int(rid)] = pred_codes

    print(f"{len(predictions)} predictions made.")
    if unresolved:
        print(f"Note: {unresolved} rooms had no id and no mapping (should be 0).")

    missing = 0
    for rid in all_ids:
        rid = int(rid)
        if rid not in predictions:
            predictions[rid] = []
            missing += 1
    print(f"Backfilled {missing} missing ids with empty predictions.")

    os.makedirs("submissions", exist_ok=True)
    test_dataset.create_submission(predictions)

if __name__ == "__main__":
    main()
