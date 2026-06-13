import os
import gdown
from pathlib import Path

MODEL_PATH = Path("final version model lstm.keras")
GDRIVE_FILE_ID = "1KZPsE6W8PV90-PREewcDNar8S5v-6aKQ"

def download():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"✅ Model already downloaded: {size_mb:.1f} MB")
        return True

    print("⬇️ Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, str(MODEL_PATH), quiet=False)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        print(f"✅ Download complete: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
        return True

    print("❌ Download failed")
    return False

if __name__ == "__main__":
    download()
