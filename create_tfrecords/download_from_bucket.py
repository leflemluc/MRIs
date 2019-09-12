from google.oauth2 import service_account
from google.cloud import storage
import nibabel as nib
import os
import pandas as pd
import tensorflow as tf
import gcsfs
import random
import numpy as np
import time

MRI_SIZE = 256
WORKING_SIZE = 168

api_key = "columbia-dl-storage-99c51286dd68.json"
credentials = service_account.Credentials.from_service_account_file(api_key)
client = storage.Client(credentials=credentials, project="columbia-dl-storage")

bucket_name = "columbia-dl-storage-bucket"
bucket = client.get_bucket(bucket_name)

prefix = "data"
blobs = bucket.list_blobs(prefix=prefix)

def download(list_names):
    
    index = 0
    couting_fail = 1
    start_100 = time.time()
    for filename in file_names_list:
        index+=1
        if couting_fail%10 == 0:
            print("couting_fail " + str(couting_fail))
        if index % 50 ==0:
            print(index);
            print('It took ' + str(time.time() - start_100))
            start_100 = time.time()
        filename = filename[5:]
        blob = bucket.blob(os.path.join(prefix, filename))
        blob.download_to_filename('data/'+filename)
        start_pixel = (MRI_SIZE - WORKING_SIZE) // 2
        nii_file = nib.load('data/'+filename).get_data()[start_pixel:start_pixel+WORKING_SIZE, start_pixel:start_pixel+WORKING_SIZE, start_pixel:start_pixel+WORKING_SIZE]
        np.save('data/'+filename[:-4], nii_file)
        os.remove('data/'+filename)
    return 

if __name__ == "__main__":
    fs = gcsfs.GCSFileSystem(token=api_key, project="columbia-dl-storage")
    with fs.open('columbia-dl-storage-bucket/ADNI_t1_list_with_fsstatus_20190111.csv') as f:
        df = pd.read_csv(f)
    df.to_csv('labels.csv')
    
    file_names_list = []
    for blob in blobs:
        file_names_list.append(blob.name)
    file_names_list= file_names_list[1:]
    download(file_names_list)
    
