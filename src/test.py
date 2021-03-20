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

def predict(review,rating):
    if rating<=3:
        text = clean_text(review)
        pred = model.predict([text])
        pred = pred[0]
        return pred

    return "The rating is already good. Don't you Worry"

# if __name__=="__main__":
#     review = input('Enter the review: ')
#     rating = int(input('Enter the rating: '))
#     print(predict(review,rating))

