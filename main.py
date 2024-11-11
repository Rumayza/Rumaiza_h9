import pandas as pd
import os
import zipfile
import requests

def download_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    zip_file_path = 'data/smsspamcollection.zip'
    
    response = requests.get(url)
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('data')  # Extract to the data folder

    df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'sms'])

    df.to_csv(os.path.join('data', 'dataset.csv'), index=False)

if __name__ == '__main__':
    download_data()
