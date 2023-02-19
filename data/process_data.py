import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets from CSV files.

    Args:
        messages_filepath (str): Filepath to the messages CSV file.
        categories_filepath (str): Filepath to the categories CSV file.

    Returns:
        pandas.DataFrame: A merged DataFrame containing the messages and categories data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left', on='id')
    return df

def clean_data(df):
    """
    Clean the input DataFrame by splitting the 'categories' column into separate
    columns for each category, converting the values to integers, dropping duplicates,
    and removing rows with category values other than 0 or 1.

    Args:
        df (pandas.DataFrame): The DataFrame to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    categories = df.categories.str.split(';', expand = True)

    row = categories.iloc[0]
    category_colnames = [col.split('-')[0] for col in row]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].astype(str).str.rpartition('-')[2]
        categories[column] = categories[column].astype(int)

    df.drop('categories', axis = 1,inplace = True)

    df = pd.concat([df,categories], axis = 1)
    df = df.drop_duplicates()
    for column in categories.columns:
        unique_values = df[column].unique()

    # I found for some reason that there was a two in four rows in the related column
    # So now all rows are dropped where there are entries unequal two one or two
    # The only risk is if the input dataset has for some reasons a lot of twos
    # a lot of rows are dropped
        if len(unique_values) > 2:
            df.drop(index = df[(df[column] != 1) & (df[column] != 0)].index, inplace = True)

    return df


def save_data(df, database_filename):
    """
    Save the input DataFrame to a SQLite database.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        database_filename (str): The filename of the database to save to.

    Returns:
        None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Clean_Data', engine, index=False)


def main():
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
