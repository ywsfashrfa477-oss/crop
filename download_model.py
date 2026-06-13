"""Model file is included directly in the repo - no download needed."""
from pathlib import Path

MODEL_PATH = Path("final version model lstm.keras")

def download():
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"✅ Model exists: {MODEL_PATH} ({size_mb:.1f} MB)")
        return True
    print(f"❌ Model file not found: {MODEL_PATH}")
    return False

if __name__ == "__main__":
    download()
