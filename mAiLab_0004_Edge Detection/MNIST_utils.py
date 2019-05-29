import numpy as np
import requests
import os
from tqdm import tqdm as tqdm
import gzip

# Return target file size in byte (int)
def check_size(url):
    r = requests.get(url, stream=True)
    return int(r.headers['Content-Length'])

# Define helper function for download (int)
def download_file(url, filename, bar=True):
    """
    Helper method handling downloading large files 
    from `url` to `filename`. Returns a pointer to `filename`.
    """
    try:
        chunkSize = 1024
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            if bar:
                pbar = tqdm(unit="B", total=check_size(url))
            for chunk in r.iter_content(chunk_size=chunkSize): 
                if chunk: # filter out keep-alive new chunks
                    if bar: 
                        pbar.update(len(chunk))
                    f.write(chunk)
        return
    except Exception as e:
        print(e)
        return

#Load image and label from xxx.gz
def read_mnist(images, labels):
    with gzip.open(labels, 'rb') as labelsFile:
        #the lable byte begin form the 0008 byte, so we set the offset to 8 for ignoring 0~7
        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)

    with gzip.open(images,'rb') as imagesFile:
        data_size = len(labels)
        # Load flat 28x28 px images
        # the lable byte begin form the 0016 byte, so we set the offset to 8 for ignoring 0~15
        features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16).reshape(data_size, 28, 28)
        
    return features, labels

#print out image by a 28*28 matrix
def print_image(file):
    for i in file:
        for j in i:
            # {:02X} output the pixel numbers by two digits hexadecimal
            # example: 255 -> FF ; 14 -> 1E
            print("{:02X}".format(j), end=' ') 
        print()
    print()
