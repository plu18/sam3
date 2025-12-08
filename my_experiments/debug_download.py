import traceback

from huggingface_hub import hf_hub_download

print("Attempting to download config.json from facebook/sam3...")
try:
    path = hf_hub_download(repo_id="facebook/sam3", filename="config.json")
    print(f"Successfully downloaded config.json to {path}")
except Exception:
    traceback.print_exc()

print("\nAttempting to download sam3.pt from facebook/sam3...")
try:
    path = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")
    print(f"Successfully downloaded sam3.pt to {path}")
except Exception:
    traceback.print_exc()
