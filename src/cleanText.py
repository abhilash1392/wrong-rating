import string 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import nltk
nltk.download('vader_lexicon')
import emoji 
import re
import spacy


def removeEmoji(text):
    return emoji.get_emoji_regexp().sub(r'',str(text))

new_words = ['app','chrome','google','apps','aap','apps','update','updated','browser']

def clean_text(text):
    text = removeEmoji(text)
    text = "".join([word.lower() for word in text if word not in string.punctuation and not word.isdigit()])
    text = word_tokenize(text)
    text = [word for word in text if word not in new_words]
    text = [WordNetLemmatizer().lemmatize(word) for word in text]
    text = " ".join(text)
    return text




