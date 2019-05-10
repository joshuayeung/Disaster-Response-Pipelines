
# import libraries
import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

import pickle

def load_data(database_filepath):
    ''' 
    
    load data from database using database_filepath 
    
    Parameters:
    database_filepath: a string contains filepath of the database

    Returns:
    X: the message column
    y: the categories
    category_name: the names of the categories

    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name='disasterresponse', con=engine)
    X = df.message.values
    remove_col = ['id', 'message', 'original', 'genre']
    y = df.loc[:, ~df.columns.isin(remove_col)]
    y.loc[:,'related'] = y['related'].replace(2,1)
    category_name = y.columns
    return X, y, category_name

url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    ''' 
    
    A tokenization function to process text data 
    
    Parameters:
    text: a string with untokenizated sentences

    Returns:
    clean_tokens: a list of tokenization words from input sentences

    '''
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
    '''
    
    A machine learning pipeline takes in the message column as input and 
    output classification results on the other 36 categories in the dataset. 

    Parameters:
    None

    Returns:
    cv: a model that uses the message column to predict classifications for 36 categories

    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' 
    
    Evaluate the model performance by f1 score, precision and recall

    Parameters:
    model: a ML model
    X_test: message from test set
    Y_test: category value from test set
    category_names: the names of the categories

    Return:
    None

    '''

    y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(data=y_pred, 
                          index=Y_test.index, 
                          columns=category_names)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    
    Save the model to a specified path
    
    Parameters:
    model: a ML model
    model_filepath: the file path that the model will be saved

    Returns:
    None

    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''

    Main function that train the classifier

    Parameters:
    arg1: the file path of the database
    arg2: the file path that the trained model will be saved

    Returns:
    None

    '''
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