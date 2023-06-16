import os
import re
import string
import pandas as pd
import joblib
import nltk
import altair as alt
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


nltk.download('punkt')
nltk.download('stopwords')


# Set the paths
models_dir = Path("/resources/models")
data_dir = Path("/resources/data")

vectorizer_path = models_dir / "tfidfvect.pkl"
models = {
    "Logistic Regression": models_dir / "Logistic_regression.pkl",
    "Random Forest": models_dir / "Random_forest.pkl",
    "SVM": models_dir / "SVM.pkl"
}
train_data_path = data_dir / "train.csv"


def load_vectorizer():
    """
    Load the TfidfVectorizer object used for vectorization.

    Returns:
        TfidfVectorizer: The loaded TfidfVectorizer object.
    """
    return joblib.load(vectorizer_path)


def load_models():
    """
    Load the pre-trained models.

    Returns:
        dict: A dictionary containing the loaded pre-trained models.
    """
    loaded_models = {}
    for model_name, model_path in models.items():
        loaded_models[model_name] = joblib.load(model_path)
    return loaded_models


def save_models(trained_models):
    """
    Save the trained models to disk.

    Args:
        trained_models (dict): A dictionary containing the trained models.
    """
    for model_name, model in trained_models.items():
        model_path = models[model_name]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)


def load_raw_data():
    """
    Load the raw training data from a CSV file.

    Returns:
        pd.DataFrame: The loaded raw training data as a DataFrame.
    """
    return pd.read_csv(train_data_path)


def string_to_list(text):
    """
    Converts a string into a list by splitting it using spaces as the delimiter.

    Args:
        text (str): The input string to be converted.

    Returns:
        list: The resulting list after splitting the string.
    """
    return text.split(' ')


def list_to_string(lst):
    """
    Converts a list into a string by joining the elements with spaces.

    Args:
        lst (list): The input list to be converted.

    Returns:
        str: The resulting string after joining the list elements.
    """
    return ' '.join(lst)

# Function to remove punctuation


def remove_punctuation(text):
    """
    Removes punctuation characters from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with punctuation removed.
    """
    return ''.join([l for l in text if l not in string.punctuation])

# Preprocessing function


def preprocess_text(text):
    """
    Preprocesses the given text by converting it to lowercase, replacing URLs and email addresses with placeholders,
    and removing punctuation.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed text.
    """

    # Convert text to lowercase
    text = text.lower()

    # Define the patterns for detecting URLs and email addresses
    pattern_url = r'http\S+'
    pattern_email = r'\S+@\S+'

    # Replace URLs and email addresses with placeholders
    subs_url = 'url-web'
    subs_email = 'email-address'
    text = re.sub(pattern_url, subs_url, text)
    text = re.sub(pattern_email, subs_email, text)

    # Remove punctuation
    text = remove_punctuation(text)

    # Return preprocessed text as a string
    return text

# Tokenize message


def tokenize_message(text):
    """
    Tokenizes the given text by splitting it into words, removing stopwords, and removing words with a length less than or equal to 2.

    Args:
        text (str): The input text.

    Returns:
        list: The tokenized words.
    """

    # Tokenization
    tokens = word_tokenize(text)
    badwords = ['email-address', 'url-web']

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords and short words
    return [word for word in tokens if word not in stop_words and len(word) > 2 and word not in badwords]


# Function to preprocess the data
def preprocess_data(data):
    # Apply text preprocessing steps
    data['processed_text'] = data['message'].apply(preprocess_text)
    data['tokenize_message_text'] = data['processed_text'].apply(
        tokenize_message)

    # Convert the message column back to string
    data['tokenize_message_list'] = data['tokenize_message_text'].apply(
        list_to_string)

    return data


def plot_model_metrics(metrics_df):
    # Generate Data
    source = pd.DataFrame(metrics_df)

    # Transform and plot data using Altair
    chart = alt.Chart(source).transform_fold(
        fold=['Trial A', 'Trial B', 'Trial C'],
        as_=['Experiment', 'Measurement']
    ).mark_bar(opacity=0.3, binSpacing=0).encode(
        x=alt.X('Measurement:Q', bin=alt.Bin(maxbins=100)),
        y=alt.Y('count()'),
        color='Experiment:N'
    )
    return chart


# Train and evaluate models
def train_and_evaluate_models(model_names, training_data, vectorizer):
    trained_models = {}
    metrics = []

    for model_name in model_names:
        if model_name == "Logistic Regression":
            model = LogisticRegression()
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear']
            }
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 5, 10]
            }
        else:
            raise ValueError("Invalid model name.")

        X_train = vectorizer.transform(
            training_data['processed_text']).toarray()
        y_train = training_data['sentiment']

        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        trained_models[model_name] = best_model

        model_path = models[model_name]
        joblib.dump(best_model, model_path)

        # Evaluate model metrics
        y_pred = best_model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred, average='macro')
        recall = recall_score(y_train, y_pred, average='macro')
        f1 = f1_score(y_train, y_pred, average='macro')

        metrics.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

    metrics_df = pd.DataFrame(metrics)

    return trained_models, metrics_df
