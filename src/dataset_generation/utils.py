import os
import random
import requests
from tqdm import tqdm
import pycld2 as cld2



def add_sampled_suffix(file_path):
    # Split the file path into directory, base filename, and extension
    directory, filename = os.path.split(file_path)
    base, ext = os.path.splitext(filename)
    
    # Add the '-sampled' suffix before the extension
    new_filename = f"{base}-sampled{ext}"
    
    # Join the directory and the new filename to get the full path
    new_file_path = os.path.join(directory, new_filename)
    
    return new_file_path


def sample_nq(input_path, output_path):
    # 307373 lines, we want 5000 of them randomly. we don't want to load all the db in memory, so we will read the file line by line and sample

    # if output_path exists, return
    if os.path.exists(output_path):
        print(f"{output_path} already exists")
        return
    
    # we take 1 line every 60 lines on average
    sampling_rate = 60
    max_samples = 5000

    # set the seed 
    random.seed(42)

    with open(input_path, "r") as f:
        with open(output_path, "w") as f_out:
            sample_count = 0
            for i, line in enumerate(f):
                if random.random() < 1 / sampling_rate:
                    f_out.write(line)
                    sample_count += 1
                if i % 100000 == 0:
                    print(f"Processed {i} lines")
                if sample_count >= max_samples:
                    break
    print("Done sampling")

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte

    print(f"Starting download from {url}...")
    with open(output_path, 'wb') as file:
        for data in tqdm(response.iter_content(block_size), total=total_size // block_size, unit='KB', unit_scale=True):
            file.write(data)
    print(f"File downloaded successfully and saved to {output_path}")


def get_first_user_content(conversation):
    for message in conversation:
        if message['role'] == 'user':
            return message['content']
    return None  # Return None or appropriate value if no user role is found

def is_english(text):
    try:
        _, _, details = cld2.detect(text)
        return details[0][1] == 'en'
    except:
        return False
