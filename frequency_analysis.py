import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from torch.utils.data import DataLoader
from dataset.hackathon import HackathonDataset
from dataset.collate import collate_fn
import numpy as np

# Load training set
dataset = HackathonDataset(split="train", download=False, seed=42)
loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

# Accumulate label frequencies
num_ops = 388
label_counts = np.zeros(num_ops)

for batch in loader:
    y = batch["Y"].numpy()  # shape (batch_size, num_ops)
    label_counts += y.sum(axis=0)

# Normalize or sort
total_labels = label_counts.sum()
frequencies = label_counts / total_labels

# Show top and bottom operations
sorted_indices = np.argsort(frequencies)
print("\n10 most common operations:")
for idx in sorted_indices[-10:][::-1]:
    print(f"Operation {idx+1}: {label_counts[idx]} occurrences ({frequencies[idx]*100:.2f}%)")

print("\n 20 rarest operations:")
for idx in sorted_indices[:20]:
    print(f"Operation {idx+1}: {label_counts[idx]} occurrences ({frequencies[idx]*100:.4f}%)")