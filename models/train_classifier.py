import sys
import sqlite3
import numpy as np
import pandas as pd
import os
from sqlalchemy import create_engine

import nltk
import re
import pickle

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    '''load data from database
    
    input:
        database_filepath: filepath of the database.
        
    output:
        X: messages.
        Y: labels.
        category_names: category name list for labels.
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MessageAndLabel',engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    '''tokenization function to process text data.
    
    input:
        text: original text messages.
        
    output:
        clean_tokens: cleaned token list after text normalizing, tokenizing and lemmatizing
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''Build pipeline model with GridSearchCV.
    
    input:
        none.
        
    output:
        cv: model structure.
    '''
    # build pipelines
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # define parameters for GridSearchCV.
    parameters = {'vect__min_df': [1, 5],
                  'clf__estimator__n_estimators':[10, 25], 
                  'clf__estimator__max_depth':[3, 5],
                  'clf__estimator__max_features': ['sqrt', 0.3]
                 }
    
    # build GridSearchCV structure.
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''performance evaluation for trained model on test datasets.
    
    input:
        model: trained model.
        X_test: message data in test set.
        Y_test: label data in test set.
        category_names: category name list for labels.
        
    output:
        printing evaluation scores.
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names, digits=2))


def save_model(model, model_filepath):
    '''Export model as a pickle file.
    
    input:
        model: trained model.
        model_filepath: output filepath.
        
    output:
        none.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()