"""Download the SigLIP2 checkpoint for the H1 replication (P7)."""

import os

from huggingface_hub import snapshot_download

TARGET = r"C:\Users\123\AppData\Local\Temp\opencode\siglip2_model"

path = snapshot_download(
    "Amirhossein75/siglip2-base-mmhs150k-fusion",
    local_dir=TARGET,
    allow_patterns=["*.json", "*.safetensors", "*.py", "*.md"],
)
print("downloaded to", path)
for name in sorted(os.listdir(TARGET)):
    size = os.path.getsize(os.path.join(TARGET, name))
    print(f"  {name:36s} {size}")
