#!/usr/bin/env python3
"""Quick test of ESMFold with a simple sequence."""

import torch
from transformers import EsmForProteinFolding

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
load_dtype = torch.float16 if device == "cuda" else torch.float32

model = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1",
    torch_dtype=load_dtype,
    low_cpu_mem_usage=True
)
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Test sequence
seq = "KFFKKLKKAVKKGFKKFAKV"
print(f"\nTesting sequence: {seq}")
print(f"Length: {len(seq)}")

try:
    with torch.no_grad():
        pdb = model.infer_pdb(seq)
    print(f"\n✅ SUCCESS! PDB output length: {len(pdb)} chars")
    print("First 300 chars of PDB:")
    print(pdb[:300])
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
