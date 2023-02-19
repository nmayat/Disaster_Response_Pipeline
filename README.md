# Disaster_Response_Pipeline
This project is a part of the Udacity Data Science Nanodegree. It aims to build a model for an API that classifies disaster messages. The project consists of three steps: an ETL pipeline, an ML pipeline, and a Flask web app.

## Project Summary
The Disaster Response Pipeline project is designed to help organizations quickly and effectively respond to natural disasters. By classifying incoming messages, emergency responders can better identify and prioritize urgent needs, saving valuable time and resources.

## Data
The project uses data from Appen (formerly Figure Eight). The messages.csv file contains the messages provided by Appen, and the corresponding labels are in the categories.csv file.

## ETL Pipeline
The ETL pipeline is responsible for cleaning the data and storing it in a SQLite database. The process_data.py script reads in the message and category data, merges the two datasets, cleans the data, and stores it in a SQLite database.

To run the ETL pipeline:
python process_data.py messages.csv categories.csv DisasterResponse.db

## ML Pipeline
The ML pipeline is responsible for training a multi-output classification model on the cleaned data. It consists of two steps: feature engineering and classification. Feature engineering involves extracting relevant features from the data using text and sentiment analysis. The extracted features are combined using a FeatureUnion. Classification involves training an XGBClassifier using the combined features. The entire pipeline is encapsulated in a Pipeline object.

To train the model, we load the data from the SQLite database, split it into training and testing sets, fit the pipeline to the training data, and use it to make predictions on the testing data. The resulting model is then saved as a pickle file.

To run the ML pipeline:
python train_classifier.py DisasterResponse.db classifier.pkl

The grid search in the pipeline might take some time to run. There is already a trained model in the models folder, so that the run.py app should work as well.

## Flask Web Appen
The Flask web app provides a user interface for entering new messages and viewing classification results. The app loads the saved model and uses it to classify new messages.

To run the Flask web app:
python run.py

The web app can than be accessed via http://http://192.168.178.30:3000

## Imbalanced Data
Imbalanced datasets can present challenges when training machine learning models, as some labels may have very few examples compared to others. In the context of this particular dataset, it means that some categories, such as "water," may have significantly fewer examples than others. This can result in the model being biased towards the majority classes, leading to poorer performance on the minority classes. The imbalance in this dataset can affect the training of the model in several ways. For example, if we use traditional metrics like accuracy, the model may perform well overall by correctly predicting the majority class but may struggle to accurately classify the minority classes. As a result, the overall accuracy may be misleading, as the model's performance on the minority classes may be much worse than on the majority classes.
In the case of disaster response messages, the consequences of misclassification can be significant. For example, misclassifying a message that requests urgent medical assistance as something less urgent could lead to delays in providing critical care, potentially resulting in loss of life. Conversely, misclassifying a message that is not urgent as something urgent could lead to unnecessary deployment of resources, which can be wasteful and hinder response efforts in other areas.
Given the potential consequences of misclassification, we may want to place a greater emphasis on recall for certain categories. For example, categories related to urgent medical assistance, such as "medical_help" and "medical_products," may require a higher recall than other categories. This means that we would prioritize correctly identifying as many instances of these categories as possible, even if it means accepting some false positives.
On the other hand, for categories that are less urgent, we may want to place a greater emphasis on precision. This means that we would prioritize correctly identifying only the instances that are truly positive, even if it means accepting some false negatives.
