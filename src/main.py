import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk 
from nltk.corpus import stopwords
import re

def pre_load_dependecies() -> None:
    nltk.download('stopwords')


def load_data() -> pd.DataFrame:
    df = pd.read_csv("../res/tripadvisor_data/reviews.csv")
    return df


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\W", " ", text)     # remove special characters
    text = re.sub(r"\s+", " ", text)    # remove extra spaces
    text = text.strip()                 # remove leading/trailing spaces
    # remove stopwords
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text


def label_sentiment(rating) -> str:
    if rating >= 3:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


def main() -> None:
    # Load the data
    df = load_data()
    
    # preview the data
    print(df.info())
    print(df.head())
    
    # Preprocess text in the dataframe
    df["cleaned_review"] = df["text"].apply(preprocess_text)
    
    
    ...
    
    
if __name__ == "__main__":
    pre_load_dependecies()
    main()