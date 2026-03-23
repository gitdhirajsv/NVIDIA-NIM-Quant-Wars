"""
Download Jane Street 19GB Data
===============================
Run this first to get the dataset.
"""

import os
import subprocess
import zipfile

DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "jane_street_data")
COMPETITION = "jane-street-real-time-market-data-forecasting"

def main():
    print("=" * 60)
    print("JANE STREET DATA DOWNLOADER")
    print("=" * 60)
    
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    print(f"\nDownloading to: {DOWNLOAD_DIR}")
    print("Size: ~19 GB (may take 10-30 minutes)\n")
    
    cmd = [
        "kaggle", "competitions", "download",
        "-c", COMPETITION,
        "-p", DOWNLOAD_DIR,
        "--force"
    ]
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in proc.stdout:
            print(line.strip())
        
        proc.wait()
        
        if proc.returncode == 0:
            print("\nDownload complete!")
            
            # Extract
            zip_file = os.path.join(DOWNLOAD_DIR, "jane-street-real-time-market-data-forecasting.zip")
            if os.path.exists(zip_file):
                print(f"\nExtracting: {zip_file}")
                with zipfile.ZipFile(zip_file, 'r') as z:
                    z.extractall(DOWNLOAD_DIR)
                print("Extracted!")
                
                # Cleanup zip
                os.remove(zip_file)
                print("Cleaned up zip file")
        else:
            print(f"\nDownload failed: {proc.returncode}")
            print("Check: API token, accepted rules, internet")
            
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
