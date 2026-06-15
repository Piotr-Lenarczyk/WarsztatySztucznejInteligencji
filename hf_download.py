# hf_download.py
import os

from huggingface_hub import snapshot_download

from pipeline.Config import PROJECT_ROOT

# Choose either the original large BART model or a highly optimized light model
REPO_ID = "facebook/bart-large-mnli"  # Size: ~1.63 GB
# REPO_ID = "typeform/distilbert-base-uncased-mnli"  # Size: ~260 MB (Alternative light option)

LOCAL_DIR = PROJECT_ROOT / "local_bart_router"

print(f"Starting isolated download for {REPO_ID} to local folder '{LOCAL_DIR}'...")
print("This separates network fetching from your Streamlit application execution.")

try:
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
        ignore_patterns=["*.bin", "*.msgpack", "*.h5"],
        token=os.environ.get("HF_TOKEN")
    )
    print("\n[SUCCESS] Model components successfully stored locally!")
    print(f"Verify the folder contents at: {LOCAL_DIR.resolve()}")
except Exception as e:
    print(f"\n[ERROR] Network connection failed or stalled: {e}")
    print("If it gets stuck here, you know it is strictly a network/proxy issue.")