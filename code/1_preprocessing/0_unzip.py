import zipfile
import os

def extract_zip(zip_path, extract_to):
    # Create the directory if it does not exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
    
    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Identify the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# # Append the relative path to the current directory
zip_path = os.path.join(current_directory, '../../raw_data/Grabaciones curadas.zip')
print(zip_path)
extract_to = os.path.join(current_directory, '../../../')

# Extract the contents of the zip file
extract_zip(zip_path, extract_to)