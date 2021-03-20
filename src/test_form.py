# Importing the libraries
import numpy as np 
import pandas as pd 
import string 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from cleanText import clean_text

model = joblib.load('../models/model.pkl')

def predict_file():
    df = pd.read_csv('../input/testfile.csv')
    df['TextClean'] = df['Text'].apply(lambda x:clean_text(x))
    df['WrongRating'] = False 
    for i,j in enumerate(df.TextClean):
        if df.loc[i,'Star']<=3:
            y_pred = model.predict([j])
            df.loc[i,'WrongRating'] = y_pred

    df.drop('TextClean',axis=1,inplace=True)
    df.to_csv('savedFile.csv',index=False)

# if __name__=="__main__":
#     predict_file()



