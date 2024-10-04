import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk 
from nltk.corpus import stopwords
import re, ast
from typing import Tuple


def pre_load_dependecies() -> None:
    nltk.download('stopwords')

        
def cleansing_data(df_offerings: pd.DataFrame, df_reviews: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ### ----- Step 1: Handling Missing Data ----- ###
    # Offerings DataFrame - Removing rows with missing 'phone' and 'details' columns since they're completely empty
    df_offerings.drop(columns=['phone', 'details'], inplace=True)
    
    # Hotel class has missing values, we'll fill missing values with the median (better than dropping)
    # df_offerings['hotel_class'].fillna(df_offerings['hotel_class'].median(), inplace=True)    # future warning with inplace=True
    df_offerings['hotel_class'] = df_offerings['hotel_class'].fillna(df_offerings['hotel_class'].median())
    
    # Reviews DataFrame - No missing values in essential columns but we'll drop rows with missing 'date_stayed'
    df_reviews.dropna(subset=['date_stayed'], inplace=True)


    ### ----- Step 2: Handling Data Types ----- ###
    # Converting date columns to datetime format for easier manipulation
    df_reviews['date'] = pd.to_datetime(df_reviews['date'], errors='coerce')
    df_reviews['date_stayed'] = pd.to_datetime(df_reviews['date_stayed'], format="%Y-%m-%d", errors='coerce')
    
    # Ensure 'offering_id' matches type with 'id' from offerings
    df_offerings['id'] = df_offerings['id'].astype(int)
    df_reviews['offering_id'] = df_reviews['offering_id'].astype(int)
    
    # Convert ratings from string format to a dictionary, and extract 'overall' rating
    df_reviews['ratings'] = df_reviews['ratings'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_reviews['overall_rating'] = df_reviews['ratings'].apply(lambda x: x.get('overall') if isinstance(x, dict) else np.nan)


    ### ----- Step 3: Removing Duplicates ----- ###
    # Dropping duplicates based on 'id' for offerings and 'id' for reviews
    df_offerings.drop_duplicates(subset=['id'], keep="first", inplace=True)
    df_reviews.drop_duplicates(subset=['id'], keep="first", inplace=True)
    
    
    ### ----- Step 4: Handling Outliers ----- ###
    # Assuming ratings should be between 0 and 5
    df_reviews = df_reviews[(df_reviews['overall_rating'] >= 0) & (df_reviews['overall_rating'] <= 5)]


    ### ----- Step 5: Feature Engineering ----- ###
    # Extract region, postal code, and locality from the address column for further analysis
    df_offerings['address'] = df_offerings['address'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_offerings['region'] = df_offerings['address'].apply(lambda x: x.get('region') if isinstance(x, dict) else np.nan)
    df_offerings['postal_code'] = df_offerings['address'].apply(lambda x: x.get('postal-code') if isinstance(x, dict) else np.nan)
    df_offerings['locality'] = df_offerings['address'].apply(lambda x: x.get('locality') if isinstance(x, dict) else np.nan)


    ### ----- Step 6: Removing Unnecessary Columns ----- ###
    # Drop columns that aren't needed for analysis
    df_offerings.drop(columns=['url', 'address'], inplace=True)
    df_reviews.drop(columns=['ratings'], inplace=True)
    
    ### ----- Step 7: Text Preprocessing ----- ###
    df_reviews["combined_text"] = df_reviews["title"] + " " + df_reviews["text"]
    
    return (df_offerings, df_reviews)



def sentiment_labeler(rating, is_optmize=False):
    if is_optmize:
        # using 1-hot encoding
        # [negative, neutral, positive]
        if rating >= 4:
            return [0, 0, 1]
        elif rating == 3:
            return [0, 1, 0]
        else:
            return [1, 0, 0]
    else:
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
        

def perform_sentiment_logistic_regression(df_reviews: pd.DataFrame) -> LogisticRegression:
    # step 1: prepare data for sentiment analysis
    # label sentiment based on overall rating (range 0-5; positive if >= 3, negative otherwise)
    print("Labeling sentiment based on overall rating...")
    df_reviews["sentiment"] = df_reviews["overall_rating"].apply(sentiment_labeler)
    
    
    # step 2: split data into training and testing sets
    x = df_reviews["combined_text"]     # features
    y = df_reviews["sentiment"]         # labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # step 3: text preprocessing and feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)
    
    
    # step 4: model selection and training
    print("Training Logistic Regression Model...")
    START_TRAIN_TIME = time.time()
    model_lr = LogisticRegression(max_iter=2000, verbose=1)
    model_lr.fit(x_train_tfidf, y_train)
    END_TRAIN_TIME = time.time() - START_TRAIN_TIME
    print(f"Training completed in {END_TRAIN_TIME:.2f} seconds.")
    
    
    # step 5: make predictions
    model_lr_predictions = model_lr.predict(x_test_tfidf)
    
    
    # step 6: evaluate model
    print("\n\nLogistic Regression Model Result: ")
    print("Classification report:", classification_report(y_test, model_lr_predictions))
    
    return model_lr
    


def main() -> None:
    START_TIME = time.time()
    # Load the data
    df_offerings = pd.read_csv("../res/tripadvisor_data/offerings.csv") 
    df_reviews = pd.read_csv("../res/tripadvisor_data/reviews.csv")
    
    # export random row to csv file before cleansing
    df_offerings.sample(5).to_csv("../output/sample/offerings_sample_before.csv", index=False)
    df_reviews.sample(5).to_csv("../output/sample/reviews_sample_before.csv", index=False)
    
    # Preprocess text in the dataframe
    print("Cleansing data...")
    df_offerings, df_reviews = cleansing_data(df_offerings, df_reviews)
    print("Data cleansing completed.")
    
    # print("offerings.csv data after cleansing")
    # print(df_offerings.info())
    # print(df_offerings.head())
    # print("="*50)
    # print("reviews.csv data after cleansing")
    # print(df_reviews.info())
    # print(df_reviews.head())
    
    # export random row to csv file after cleansing
    df_offerings.sample(5).to_csv("../output/sample/offerings_sample_after.csv", index=False)
    df_reviews.sample(5).to_csv("../output/sample/reviews_sample_after.csv", index=False)
    
    print(f"df_offerings [{df_offerings.shape[0]}] row x [{df_offerings.shape[1]}] cols")
    print(f"df_reviews [{df_reviews.shape[0]}] row x [{df_reviews.shape[1]}] cols")
    print(f"Time taken for data loading, cleansing, and preparation: {time.time() - START_TIME:.2f} seconds")
    
    model = perform_sentiment_logistic_regression(df_reviews)
    
    ...
    
    
if __name__ == "__main__":
    pre_load_dependecies()
    main()