# download_raw_data.py
# ─────────────────────────────────────────────────────────────────────────────
# Downloads historical raw data from Google Drive into data/raw/.
#
# Run this ONCE before running the historical pipeline for the first time.
# GitHub Actions runs this automatically before the live pipeline.
#
# Setup:
#   1. Upload your data/raw/ folder to Google Drive
#   2. Right-click folder → Share → "Anyone with the link" → copy the link
#   3. Add DRIVE_RAW_FOLDER_ID to your .env file (see below)
#
# Get the folder ID from the Drive link:
#   https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQr  ← this part
#                                           ^^^^^^^^^^^^^^^^^^^^
#
# .env entry:
#   DRIVE_RAW_FOLDER_ID=1AbCdEfGhIjKlMnOpQr
#
# How to run:
#   pip install gdown
#   python3 download_raw_data.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

try:
    import gdown
except ImportError:
    print("gdown not installed. Run:  pip install gdown")
    sys.exit(1)

FOLDER_ID  = os.getenv("DRIVE_RAW_FOLDER_ID", "")
OUTPUT_DIR = ROOT / "data" / "raw"

print("=" * 60)
print("DOWNLOAD RAW DATA — Google Drive → data/raw/")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

if not FOLDER_ID:
    print("\n❌  DRIVE_RAW_FOLDER_ID not set.")
    print("   Add it to your .env file:")
    print("   DRIVE_RAW_FOLDER_ID=your_folder_id_here")
    print("\n   Get it from your Drive link:")
    print("   https://drive.google.com/drive/folders/<THIS_PART>")
    sys.exit(1)

# If data/raw already has files, skip to avoid re-downloading
existing = list(OUTPUT_DIR.glob("*.json")) + list(OUTPUT_DIR.glob("*.csv"))
if existing:
    print(f"\n✅  data/raw/ already has {len(existing)} files — skipping download.")
    print("   Delete data/raw/ and re-run to force a fresh download.")
    sys.exit(0)

print(f"\nFolder ID : {FOLDER_ID}")
print(f"Output    : {OUTPUT_DIR}")
print("\nDownloading... (this may take a few minutes for 849MB)\n")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# gdown downloads the folder contents into a subfolder named after the Drive folder.
# We use a temp dir then move files up one level into data/raw/.
TEMP_DIR = ROOT / "data" / "_raw_download_tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

try:
    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    gdown.download_folder(url, output=str(TEMP_DIR), quiet=False, use_cookies=False)

    # Move downloaded files into data/raw/
    downloaded = 0
    for item in TEMP_DIR.rglob("*"):
        if item.is_file():
            dest = OUTPUT_DIR / item.name
            shutil.move(str(item), str(dest))
            downloaded += 1
            print(f"  → {item.name}")

    # Clean up temp dir
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    print(f"\n✅  Downloaded {downloaded} files → {OUTPUT_DIR}")

except Exception as e:
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print(f"\n❌  Download failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure the Drive folder is shared: 'Anyone with the link'")
    print("  2. Check that DRIVE_RAW_FOLDER_ID is correct")
    print("  3. Try: gdown --folder https://drive.google.com/drive/folders/YOUR_ID")
    sys.exit(1)
