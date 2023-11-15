import requests
import os
from tqdm import tqdm

# URL of the file to be downloaded
url = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_S.gguf?download=true'

# Path where the file will be saved
save_path = os.path.join('./Models/mistral', 'mistral-7b-instruct-v0.1.Q4_K_S.gguf')

# Send HTTP request to the URL
response = requests.get(url, stream=True)

# Get the total size of the file
total_size = int(response.headers.get('content-length', 0))

# Initialize a progress bar
progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

# Ensure the request was successful
if response.status_code == 200:
    # Write the content of the response to a file in the specified directory
    with open(save_path, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            # Update the progress bar
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
else:
    print(f'Failed to download file: {response.status_code}')
