import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')

import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pickle

def load_data(database_filepath):
    """
    Load the clean data from a SQLite database and split it into the message text
    (X), the category labels (y), and the names of the category columns.

    Args:
        database_filepath (str): The filepath of the database to load from.

    Returns:
        tuple: A tuple containing the message text (X), the category labels (y),
        and the names of the category columns as a pandas DataFrame.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Clean_Data', engine)

    X = df.message
    y = df.drop(['id', 'message','original', 'genre'], axis = 1)
    category_names = y.columns
    return X,y,category_names


def tokenize(text):
    """
    Tokenize the input text by splitting it into individual words, lemmatizing each word,
    and returning a list of the cleaned tokens.

    Args:
        text (str): The text to be tokenized.

    Returns:
        list: A list of the cleaned and lemmatized tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class SentimentExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer that extracts sentiment scores from text using the
    SentimentIntensityAnalyzer from the NLTK library.

    Attributes:
        analyzer (SentimentIntensityAnalyzer): An instance of the SentimentIntensityAnalyzer
            class used to compute sentiment scores.

    Methods:
        sentiment_score(text):
            Computes the sentiment score for a given text using the SentimentIntensityAnalyzer.
        fit(x, y=None):
            Returns self.
        transform(X):
            Applies the sentiment_score function to each element in X and returns a
            pandas DataFrame with the computed sentiment scores.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def sentiment_score(self, text):
        """
        Computes the sentiment score for a given text using the SentimentIntensityAnalyzer.

        Args:
            text (str): The text for which the sentiment score is to be computed.

        Returns:
            float: The sentiment score for the given text.
        """
        sentiment = self.analyzer.polarity_scores(text)
        return sentiment['compound']

    def fit(self, x, y=None):
        """
        Returns self.

        Args:
            x: The input data.
            y: The target data.

        Returns:
            SentimentExtractor: Returns self.
        """
        return self

    def transform(self, X):
        """
        Applies the sentiment_score function to each element in X and returns a
        pandas DataFrame with the computed sentiment scores.

        Args:
            X (pandas.DataFrame): The input data.

        Returns:
            pandas.DataFrame: A DataFrame with the computed sentiment scores.
        """
        sentiment_scores = X.apply(self.sentiment_score)
        return pd.DataFrame(sentiment_scores)

def build_model():
    """
    Builds a machine learning model using a pipeline with text and sentiment features,
    and an XGBoost classifier with multi-output classification.

    Returns:
        GridSearchCV: A GridSearchCV object that can be used to fit the model to the data
        and find the best hyperparameters.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('sentiment_pipeline', SentimentExtractor())
        ])),
        ('clf', MultiOutputClassifier(XGBClassifier()))
    ])

    parameters = {
        #'features__text_pipeline__vect__max_df': [0.5, 0.75],
        #'features__text_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
        #'features__text_pipeline__tfidf__use_idf': [True, False],
        'clf__estimator__max_depth': [3],#[3, 5, 7],
        #'clf__estimator__n_estimators': [50, 100, 150],
        #'clf__estimator__learning_rate': [0.01, 0.1, 0.5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the input model on the test set and print a
    classification report for each category.

    Args:
        model (sklearn.pipeline.Pipeline): The model to be evaluated.
        X_test (pandas.DataFrame): The test features.
        Y_test (pandas.DataFrame): The test target variable.
        category_names (list): A list of the category names.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred,columns = Y_test.columns)
    for col in Y_test.columns:
        print(col)
        print(classification_report(Y_test[col], y_pred[col]))


def save_model(model, model_filepath):
    """
    Save the input model as a binary file using the pickle library.

    Args:
        model (object): The model to be saved.
        model_filepath (str): The filepath (including filename) for the saved model.

    Returns:
        None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    The main function that loads the data, trains a machine learning model, evaluates
    the model, and saves the model to a pickle file.

    Args:
        None

    Returns:
        None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        print(model.best_params_)

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
