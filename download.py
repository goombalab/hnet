
import os
from huggingface_hub import snapshot_download

# For faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# The repository to download from
repo_id = "cartesia-ai/hnet_2stage_XL"

# The directory to save the files to
local_dir = "./data"

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading files from {repo_id} to {local_dir}...")

# Download the repository
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False, # Set to False to download the actual files
)

print("Download complete.")
