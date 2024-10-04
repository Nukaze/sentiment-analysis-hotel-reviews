import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk 
from nltk.corpus import stopwords
import re, ast
from typing import Tuple


def pre_load_dependecies() -> None:
    nltk.download('stopwords')


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


def cleansing_data(df_offerings, df_reviews) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ### ----- Step 1: Handling Missing Data ----- ###
    # Offerings DataFrame - Removing rows with missing 'phone' and 'details' columns since they're completely empty
    df_offerings.drop(columns=['phone', 'details'], inplace=True)
    
    # Hotel class has missing values, we'll fill missing values with the median (better than dropping)
    df_offerings['hotel_class'].fillna(df_offerings['hotel_class'].median(), inplace=True)

    # Reviews DataFrame - No missing values in essential columns but we'll drop rows with missing 'date_stayed'
    df_reviews.dropna(subset=['date_stayed'], inplace=True)


    ### ----- Step 2: Handling Data Types ----- ###
    # Converting date columns to datetime format for easier manipulation
    df_reviews['date'] = pd.to_datetime(df_reviews['date'], errors='coerce')
    df_reviews['date_stayed'] = pd.to_datetime(df_reviews['date_stayed'], errors='coerce')
    
    # Ensure 'offering_id' matches type with 'id' from offerings
    df_offerings['id'] = df_offerings['id'].astype(int)
    df_reviews['offering_id'] = df_reviews['offering_id'].astype(int)

    # Convert ratings from string format to a dictionary, and extract 'overall' rating
    df_reviews['ratings'] = df_reviews['ratings'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_reviews['overall_rating'] = df_reviews['ratings'].apply(lambda x: x.get('overall') if isinstance(x, dict) else np.nan)


    ### ----- Step 3: Removing Duplicates ----- ###
    # Dropping duplicates based on 'id' for offerings and 'id' for reviews
    df_offerings.drop_duplicates(subset=['id'], inplace=True)
    df_reviews.drop_duplicates(subset=['id'], inplace=True)
    
    
    ### ----- Step 4: Handling Outliers ----- ###
    # Assuming ratings should be between 1 and 5
    df_reviews = df_reviews[(df_reviews['overall_rating'] >= 1) & (df_reviews['overall_rating'] <= 5)]


    ### ----- Step 5: Feature Engineering ----- ###
    # Extract region, postal code, and locality from the address column for further analysis
    df_offerings['address'] = df_offerings['address'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_offerings['region'] = df_offerings['address'].apply(lambda x: x.get('region') if isinstance(x, dict) else np.nan)
    df_offerings['postal_code'] = df_offerings['address'].apply(lambda x: x.get('postal-code') if isinstance(x, dict) else np.nan)
    df_offerings['locality'] = df_offerings['address'].apply(lambda x: x.get('locality') if isinstance(x, dict) else np.nan)


    ### ----- Step 6: Removing Unnecessary Columns ----- ###
    # Drop columns that aren't needed for analysis
    df_offerings.drop(columns=['url', 'address'], inplace=True)
    
    return (df_offerings, df_reviews)



def main() -> None:
    # Load the data
    df_offerings = pd.read_csv("../res/tripadvisor_data/offerings.csv") 
    df_reviews = pd.read_csv("../res/tripadvisor_data/reviews.csv")
    
    # preview the data
    print("offerings.csv data")
    print(df_offerings.info())
    print(df_offerings.head())
    print("="*50)
    print("reviews.csv data")
    # found the data are corrupted, the rows are loss from 2m -> 878,561
    # so need to fix the data
    print(df_reviews.info())
    print(df_reviews.head())
    
    # export random row to csv file before cleansing
    df_offerings.sample(5).to_csv("../output/sample/offerings_sample_before.csv", index=False)
    df_reviews.sample(5).to_csv("../output/sample/reviews_sample_before.csv", index=False)
    
    # Preprocess text in the dataframe
    df_offerings, df_reviews = cleansing_data(df_offerings, df_reviews)
    print("="*50)
    print("="*50)
    print("="*50)
    print("offerings.csv data after cleansing")
    print(df_offerings.info())
    print(df_offerings.head())
    print("="*50)
    print("reviews.csv data after cleansing")
    print(df_reviews.info())
    print(df_reviews.head())
    
    # export random row to csv file after cleansing
    df_offerings.sample(5).to_csv("../output/sample/offerings_sample_after.csv", index=False)
    df_reviews.sample(5).to_csv("../output/sample/reviews_sample_after.csv", index=False)
    
    
    ...
    
    
if __name__ == "__main__":
    pre_load_dependecies()
    main()