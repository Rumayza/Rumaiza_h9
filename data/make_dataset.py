import pandas as pd
import os
import zipfile
import requests

def download_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    
    # Download the ZIP file
    response = requests.get(url)
    zip_path = 'data/smsspamcollection.zip'
    
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')

    # Load the dataset
    df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'sms'])
    
    # Save the dataset to a CSV file
    df.to_csv('data/dataset.csv', index=False)

if __name__ == '__main__':
    download_data()
