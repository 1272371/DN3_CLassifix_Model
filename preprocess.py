import os
import re
import string
import pandas as pd
import joblib
import streamlit as st
import altair as alt
import xgboost as xgb
import altair as alt
import logging


from math import ceil
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder

# Create a logger object
logger = logging.getLogger(__name__)
# Configure the logger to write to a file
log_file = "training.log"

# nltk.download('punkt')
# nltk.download('stopwords')

# Set the paths
models_dir = Path("resources/models")
data_dir = Path("resources/data")
vectorizer_dir = Path("resources/models/")

vectorizer_path = models_dir / f"vectorizer_tfid.pkl" 
train_data_path = data_dir / "train.csv"
matrics_df_path = data_dir / "metrics_df.csv"

model_names = ["Logistic Regression", "Random Forest",
               "Support Vector Machine", "K-Nearest Neighbors"]

models = {
    "Logistic Regression":"logistic_regression_model",
    "Random Forest": "random_forest_model",
    "Support Vector Machine":"support_vector_machine_model",
    "K-Nearest Neighbors":"k-nearest_neighbors_model"
}

logging.basicConfig(filename=data_dir / log_file, level=logging.INFO)


def display_vect_metrics(vectorizer, data, st):
    """
    Displays various metrics and information about the fitted TfidfVectorizer.

    Args:
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        data (pd.DataFrame): The preprocessed data used for fitting the vectorizer.
    """
    vocab_size = len(vectorizer.vocabulary_)
    st.write("Vocabulary Size:", vocab_size)

    idf_values = vectorizer.idf_
    st.write("IDF Values:", idf_values)

    document_term_matrix = vectorizer.transform(data)
    st.write("Document-Term Matrix:")
    for i in range(document_term_matrix.shape[0]):
        row = document_term_matrix.getrow(i)
        features = row.indices
        tfidf_values = row.data
        st.write("Document", i, ":")
        for feature, tfidf in zip(features, tfidf_values):
            st.write(f"Feature {feature}: TF-IDF {tfidf}")

    # features = vectorizer.get_feature_names()
    # st.write("Features:", features)

def train_vectorizer(data,features=200000):
    """
    Trains a TF-IDF vectorizer using the provided data.

    Args:
        data (pd.DataFrame): The training data.

    Returns:
        TfidfVectorizer: The trained TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=features)
    vectorizer.fit(data)
    return vectorizer

def save_vectorizer(vectorizer, vectorizer_path):
    """
    Saves the trained vectorizer to a file.

    Args:
        vectorizer (TfidfVectorizer): The trained TF-IDF vectorizer.
        vectorizer_path (str): The path to save the vectorizer file.
    """
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)

def load_vectorizer():
    """
    Load the TfidfVectorizer object used for vectorization.

    Returns:
        TfidfVectorizer: The loaded TfidfVectorizer object, or None if not found or permission denied.
    """
    try:
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError:
        vectorizer = None
    except PermissionError:
        print("Permission denied: Unable to access the vectorizer file.")
        vectorizer = None

    if vectorizer is None:
        print("Vectorizer not found. Training a new vectorizer...")
        data = load_raw_data()
        preprocessed_data = preprocess_data(data)
        vectorizer = train_vectorizer(preprocessed_data['processed_text'])
        save_vectorizer(vectorizer, vectorizer_path)
        print("New vectorizer trained and saved.")
    return vectorizer

def save_trained_models(models):
    """
    Saves the trained models to files.

    Args:
        models (dict): A dictionary containing the trained models.
        models_dir (Path): The directory to save the trained models.
    """
    os.makedirs(models_dir, exist_ok=True)
    for model_name, model in models.items():
        model_path = models_dir / f"{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_path)

def load_model(model_name = ""):
    """
    Load the TfidfVectorizer object used for vectorization.

    Returns:
        TfidfVectorizer: The loaded TfidfVectorizer object, or None if not found or permission denied.
    """
    try:
        model_path = models_dir / (str(models[model_name]) + '.pkl')
        model = joblib.load(model_path)
    except FileNotFoundError:
        model = None
        print(f"MODEL {model_name} NOT FOUND!")
    except PermissionError:
        print("Permission denied: Unable to access the model file.")
        model = None
    return model

def load_raw_data(raw_data_path=train_data_path):
    """
    Load the raw training data from a CSV file.

    Args:
        raw_data (str): The file path of the CSV file.

    Returns:
        pd.DataFrame: The loaded raw training data as a DataFrame, or an empty DataFrame if not found.
    """
    try:
        return pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"File not found: {raw_data_path}")
        return pd.DataFrame()  # Return an empty DataFrame
    except Exception as e:
        print(f"An error occurred while loading the data: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame

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

def remove_punctuation(text):
    """
    Removes punctuation characters from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with punctuation removed.
    """
    return ''.join([l for l in text if l not in string.punctuation])

def preprocess_text(text):
    """
    Preprocesses the given text by converting it to lowercase, replacing URLs and email addresses with placeholders,
    and removing punctuation.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed text.
    """
    text = text.lower()
    pattern_url = r'http\S+'
    pattern_email = r'\S+@\S+'
    subs_url = 'url-web'
    subs_email = 'email-address'
    text = re.sub(pattern_url, subs_url, text)
    text = re.sub(pattern_email, subs_email, text)
    text = remove_punctuation(text)
    return text

def tokenize_message(text):
    """
    Tokenizes the given text by splitting it into individual words.

    Args:
        text (str): The input text.

    Returns:
        list: The list of tokens.
    """
    return word_tokenize(text)

def remove_stopwords(tokens):
    """
    Removes stopwords from the given list of tokens.

    Args:
        tokens (list): The input list of tokens.

    Returns:
        list: The list of tokens after removing stopwords.
    """
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in stop_words]

def preprocess_data(data):
    """
    Preprocesses the given data by applying text preprocessing steps to the 'message' column.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    data['processed_text'] = data['message'].apply(preprocess_text)
    data['processed_text'] = data['processed_text'].apply(tokenize_message)
    data['processed_text'] = data['processed_text'].apply(remove_stopwords)
    data['processed_text'] = data['processed_text'].apply(list_to_string)
    data['tokenized_message'] = data['processed_text'].apply(tokenize_message)
    data['tokenized_message'] = data['tokenized_message'].apply(remove_stopwords)
    return data

def train_models(model_names, training_data, vectorizer, split_ratio):
    """
    Trains multiple classification models using the provided training data and TF-IDF vectorizer.

    Args:
        model_names (list): A list of model names to train.
        training_data (pd.DataFrame): The training data.
        vectorizer (TfidfVectorizer): The trained TF-IDF vectorizer.
        split_ratio (float): The ratio for train-test split.

    Returns:
        dict: A dictionary containing the trained models.
        pd.DataFrame: A DataFrame containing the evaluation metrics for each model.
    """

    trained_models = {}
    metrics = []
    for model_name in model_names:
        logger.info(f"Training model: {model_name}")
        if model_name == "Logistic Regression":
            model = LogisticRegression()
            param_grid = {
                'C': [0.1, 1.0, 5.0],
                'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
            }
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 50, 100]
            }
        elif model_name == "Support Vector Machine":
            model = SVC()
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
        elif model_name == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
            param_grid = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        else:
            raise ValueError("Invalid model name.")

        X_train = vectorizer.transform(training_data['processed_text']).toarray()
        y_train = training_data['sentiment']
        print(training_data['processed_text'])
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42, test_size=split_ratio)

        logger.info(f"Performing grid search for {model_name}...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        trained_models[model_name] = best_model
        logger.info(f"Training complete for {model_name}")

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        metrics.append([model_name, accuracy, precision, recall, f1])

    metrics_df = pd.DataFrame(metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    metrics_df.to_csv(matrics_df_path, index=False)

    logger.info("Training and evaluation complete for all models")
    return trained_models, metrics_df

def generate_metrics_chart(metrics_df):
    """
    Generates bar charts to visualize the evaluation metrics of the trained models.

    Args:
        metrics_df (pd.DataFrame): The DataFrame containing the evaluation metrics.

    Returns:
        None
    """
    num_metrics = len(metrics_df.columns) - 1  # Exclude the 'Model' column
    num_rows = ceil(num_metrics / 2)

    chart_config = {
        'title': 'Model Evaluation Metrics',
        'width': 150,
        'height': 100
    }

    charts = []
    for i, metric in enumerate(metrics_df.columns[1:], start=1):
        row = (i - 1) // 2  # Calculate the row index (0-based)
        col = (i - 1) % 2   # Calculate the column index (0 or 1)

        chart = alt.Chart(metrics_df).mark_bar().encode(
            x='Model',
            y=alt.Y(metric, axis=alt.Axis(title=metric)),
            color='Model'
        ).properties(**chart_config)

        if col == 0:
            charts.append([chart])
        else:
            charts[row].append(chart)

    vconcat_charts = [alt.hconcat(*row) for row in charts]
    combined_chart = alt.vconcat(*vconcat_charts)
    st.write(combined_chart)

def most_common_word_plot():
    data = load_raw_data()
    data = preprocess_data(data)
    sentiment_mapping = {-1: 'Anti', 0: 'Neutral', 1: 'Pro', 2: 'News'}
    sentiment_mapped  = {}
    data['mapped_sentiment'] = data['sentiment'].map(sentiment_mapping)
    st.info('Sample Data')
    st.write(data[["tokenized_message","mapped_sentiment"]].head(3))
    for sentiment in sentiment_mapping.values():
        sentiment_mapped[sentiment] = list(data[data['mapped_sentiment'] == sentiment]['tokenized_message'].explode())
    return plot_common_words(sentiment_mapped)

def plot_common_words(sentiment_mapped):
    """
    Plots the most common words per sentiment category.

    Args:
        sentiment_mapped (dict): The dictionary mapping sentiment categories to corresponding word lists.

    Returns:
        altair.vegalite.v4.api.Chart: The stacked bar chart visualization.
    """
    # Create an empty list to store word frequency DataFrames
    word_freq_list = []

    # Iterate over each sentiment category and its corresponding word list
    for sentiment, words in sentiment_mapped.items():
        # Count the frequency of each word
        word_counts = pd.Series(words).value_counts().reset_index()
        word_counts.columns = ['Word', 'Frequency']
        # Add the sentiment category to the DataFrame
        word_counts['Sentiment'] = sentiment
        # Append the word frequencies to the list
        word_freq_list.append(word_counts)

    # Concatenate the word frequency DataFrames
    word_freq = pd.concat(word_freq_list, ignore_index=True)

    # Sort the DataFrame by frequency in descending order
    word_freq.sort_values(by=['Sentiment', 'Frequency'], ascending=[True, False], inplace=True)

    # Select the top N most common words per sentiment category
    top_words = word_freq.groupby('Sentiment').head(10)

    chart_config = {
        'title': 'Top 10 Most Common Words per Sentiment Category',
        'width': 800,
        'height': 600
    }
    # Create a stacked bar chart using Altair
    chart = alt.Chart(top_words).mark_bar().encode(
        x=alt.X('Word', sort='-y'),
        y='Frequency',
        color='Sentiment',
        tooltip=['Word', 'Frequency', 'Sentiment']
    ).properties(**chart_config).configure_mark(
        opacity=0.8
    ).configure_legend(
        orient='right'
    )

    return chart

def plot_xgb(df):
    # Preprocess the messages and create a feature matrix using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['processed_text'])

    # Define the target variable
    y = df['mapped_sentiment']

    # Encode the sentiment labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"{y_encoded=}")
    # Train an XGBoost model
    model = xgb.XGBClassifier()
    model.fit(X, y_encoded)

    # Get the feature importance from the XGBoost model
    feature_names = vectorizer.get_feature_names_out()
    feature_importances = model.feature_importances_

    # Create a dictionary mapping feature names to importance scores
    feature_scores = {feature_names[i]: score for i, score in enumerate(feature_importances)}

    # Sort the feature scores in descending order
    sorted_scores = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    # Generate word cloud based on the feature scores
    wordcloud = WordCloud(width=600, height=400, background_color='white')
    wordcloud.generate_from_frequencies(dict(sorted_scores))

    # Display the word cloud using Streamlit
    st.image(wordcloud.to_image())
    st.title('Word Cloud of Messages (XGBoost)')

def distribution_per_sentiment(sentiment_mapped):
    # Create an empty list to store charts for each sentiment
    charts = []
    
    # Iterate over each sentiment category and its corresponding word list
    for sentiment, words in sentiment_mapped.items():
        # Count the frequency of each word
        word_counts = pd.Series(words).value_counts().reset_index()
        word_counts.columns = ['Word', 'Frequency']
        
        # Create an Altair chart for the current sentiment
        chart = alt.Chart(word_counts).mark_bar().encode(
            x='Word',
            y='Frequency',
            color='Sentiment'
        ).properties(
            title=f'Distribution of Words per Sentiment: {sentiment}'
        )
        
        # Add the chart to the list
        charts.append(chart)
    
    # Combine all charts into a single Altair chart
    combined_chart = alt.vconcat(*charts)
    
    # Display the chart
    return combined_chart