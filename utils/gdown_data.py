import os
import sys
import subprocess

def install_gdown():
    """Install gdown if it's not already installed."""
    try:
        import gdown
        print("gdown is already installed.")
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        print("gdown has been installed successfully.")

def download_from_gdrive(url, output_dir=None):
    """Download files from Google Drive folder."""
    import gdown

    # Handle single-file URLs
    if "/file/d/" in url:
        # Extract the file ID from the URL
        file_id = url.split("/file/d/")[1].split("/")[0]
        # Determine output directory
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        # Construct direct download URL
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        # Set output filename
        output_path = os.path.join(output_dir, "redhair.zip")
        print(f"Downloading file with ID: {file_id} to {output_path}...")
        gdown.download(download_url, output_path, quiet=False)
        print("File downloaded successfully.")
        return True
    
    # Extract the folder ID from the URL
    if "folders" in url:
        folder_id = url.split("folders/")[1].split("?")[0]
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_data")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Downloading folder with ID: {folder_id} to {output_dir}...")
        try:
            # Try to download the entire folder first
            try:
                gdown.download_folder(url=url, output=output_dir, quiet=False)
                print(f"Files downloaded successfully to {output_dir}")
                return True
            except Exception as e:
                error_msg = str(e)
                # Check if the error is due to the 50-file limit
                if "more than 50 files" in error_msg:
                    print("The folder has more than 50 files. Attempting to download files individually...")
                    return download_large_folder(folder_id, output_dir)
                else:
                    print(f"Error downloading files: {error_msg}")
                    return False
        except Exception as e:
            print(f"Error downloading files: {str(e)}")
            return False
    else:
        print("The provided URL doesn't seem to be a Google Drive folder URL.")
        return False

def download_large_folder(folder_id, output_dir):
    """Handle folders with more than 50 files by downloading files in batches."""
    import gdown
    import requests
    import re
    from bs4 import BeautifulSoup
    
    try:
        # First, try to get folder contents via API
        print("Getting folder contents...")
        
        # Use direct URL to access folder
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        session = requests.Session()
        
        # Get folder page to extract file IDs
        response = session.get(folder_url)
        if response.status_code != 200:
            print(f"Error accessing folder: HTTP status {response.status_code}")
            return False
        
        # Look for file IDs in the page content
        file_ids = re.findall(r'"(https://drive\.google\.com/file/d/([^/]+)/view[^"]*)"', response.text)
        
        if not file_ids:
            print("Could not find any files in this folder using direct method.")
            return False
        
        print(f"Found {len(file_ids)} files. Starting download...")
        
        # Download each file individually
        success_count = 0
        for idx, (file_url, file_id) in enumerate(file_ids):
            try:
                output_file = os.path.join(output_dir, f"file_{file_id}")
                # Convert from file URL to download URL
                download_url = f"https://drive.google.com/uc?id={file_id}"
                print(f"[{idx+1}/{len(file_ids)}] Downloading file {file_id}...")
                gdown.download(download_url, output_file, quiet=False)
                success_count += 1
            except Exception as e:
                print(f"Error downloading file {file_id}: {str(e)}")
                
        print(f"Download completed. Successfully downloaded {success_count} out of {len(file_ids)} files.")
        return success_count > 0
    
    except Exception as e:
        print(f"Error in batch download: {str(e)}")
        return False

if __name__ == "__main__":
    # Install gdown if needed
    install_gdown()
    
    # URL of the Google Drive folder
    gdrive_url = "https://drive.google.com/drive/folders/10qYd-3ASl7uLqCCmjhJ5IWVoiMmSJ44D"
    
    # Download the files
    download_from_gdrive(gdrive_url)




