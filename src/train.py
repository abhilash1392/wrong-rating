# Importing the libraries
from nltk import tokenize
from nltk.util import ngrams
import numpy as np 
import pandas as pd 
import string 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from cleanText import clean_text
from sklearn.metrics import accuracy_score


def run():
    df = pd.read_csv('../input/chrome_reviews.csv')
    df = df[['Text','Star']]
    df['TextClean'] = df['Text'].apply(lambda x: clean_text(x))
    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df["TextClean"].apply(lambda x:sid.polarity_scores(x))
    df = pd.concat([df.drop(['sentiment'],axis=1),df['sentiment'].apply(pd.Series)],axis=1)
    PositiveReview = []
    for i,j in enumerate(df.pos):
        if j>=0.4:
            PositiveReview.append(True)
        else:
            PositiveReview.append(False)

    print(sid.polarity_scores(clean_text('This is just awesome')))


    df['positiveReview'] = PositiveReview
    cvt = CountVectorizer(tokenizer=word_tokenize)

    clf = RandomForestClassifier()
    pipe = make_pipeline(cvt,clf)
    pipe.fit(df['TextClean'],df['positiveReview'])
    joblib.dump(pipe,"../models/model.pkl")



if __name__=="__main__":
    run()

