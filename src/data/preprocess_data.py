import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['sms'] = df['sms'].apply(lambda x: x.lower())
    df['sms'] = df['sms'].apply(word_tokenize)
    return df
