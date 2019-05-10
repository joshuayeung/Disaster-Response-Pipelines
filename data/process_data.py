# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''

    Load messages and categories dataset and merge the two datasets

    Parameters:
    messages_filepath: the file path of the messages dataset
    categories_filepath: the file path of the categories dataset

    Returns:
    df: a dataframe that merges the two datasets

    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, how='inner', on='id')
    
    return df

def clean_data(df):

    '''
    
    Clean the dataframe

    Parameters:
    df: unclean dataframe

    Returns:
    df: cleaned dataframe

    '''

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from 'df'
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    ''' Save the data (df) to database (database_filename) '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disasterresponse', engine, index=False, if_exists='replace')


def main():
    '''

    Main function that process the datasets

    Parameters:
    arg1: the file path of the messages dataset
    arg2: the file path of the categories dataset
    arg3: the database path that the data will be saved

    Returns:
    None

    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()