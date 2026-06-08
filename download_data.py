import os
import subprocess
import zipfile

# Downloads Titanic data from Kaggle and extracts it securely
def download_titanic_data(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'train.csv')
    
    if os.path.exists(csv_path):
        print(f"Dataset already exists at {output_dir}. Skipping download.")
        return True

    home_dir = os.path.expanduser('~') # Returns the user folder- For example on Windows: C:\Users\Yarin
    kaggle_dir = os.path.join(home_dir, '.kaggle')
    
    # Checking permission types:
    # Method 1 - Checks if exists: .kaggle/kaggle.json - This is the old and common login file.
    has_json = os.path.exists(os.path.join(kaggle_dir, 'kaggle.json'))
    
    # Method 2 - Checks for: .kaggle/access_token - A newer authentication method.
    has_access_token = os.path.exists(os.path.join(kaggle_dir, 'access_token'))
    
    # Method 3 - Checks for environment variables: KAGGLE_USERNAME, KAGGLE_KEY
    has_env_old = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    
    # Method 4 - New Token - Checks whether there is: KAGGLE_API_TOKEN
    has_env_new = os.environ.get("KAGGLE_API_TOKEN")
    
    # If there are no permissions (none of the methods were found)
    if not (has_json or has_access_token or has_env_old or has_env_new):
        print("\n" + "="*60)
        print("ERROR: Kaggle Credentials Missing!")
        print(f"Please ensure your token is configured inside: {kaggle_dir}")
        print("Secrets are not hardcoded in this script to prevent data leakage.")
        print("="*60 + "\n")
        return False
    
    print("Connecting to Kaggle and downloading Titanic data securely...")
    try:
        # Allows to run Terminal commands from within Python. If the command fails, an Exception will be thrown
        subprocess.run(['kaggle', 'competitions', 'download', '-c', 'titanic', '-p', output_dir], check=True)
        
        # If a zip downloaded - extract the files
        zip_path = os.path.join(output_dir, 'titanic.zip')
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path)  
            print("Data downloaded and extracted successfully.")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"Kaggle CLI command failed: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return False