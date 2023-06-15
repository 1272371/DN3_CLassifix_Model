import streamlit as st
import joblib
import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Define file paths
models_dir = Path("resources/models")
data_dir = Path("resources/data")
vectorizer_path = models_dir / "tfidfvect.pkl"
models = {
    "Logistic Regression": models_dir / "Logistic_regression.pkl",
    "Random Forest": models_dir / "Random_forest.pkl"
}
train_data_path = data_dir / "train.csv"

# Load the vectorizer
def load_vectorizer():
    return joblib.load(vectorizer_path)

# Load the pre-trained models
def load_models():
    loaded_models = {}
    for model_name, model_path in models.items():
        loaded_models[model_name] = joblib.load(model_path)
    return loaded_models

# Save the trained models
def save_models(trained_models):
    for model_name, model in trained_models.items():
        model_path = models[model_name]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

# Load the raw data
def load_raw_data():
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

import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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
    text = re.sub(r'\W+', '', text)

    # Remove web-urls
    text = re.sub(pattern_url, subs_url, text)

    # Return preprocessed text as a string
    return remove_punctuation(text)

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

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords and punctuation
    return [word for word in tokens if word not in stop_words and len(word) > 2]


# Function to preprocess the data
def preprocess_data(data):
    # Convert the message column from string to list
    data['processed_message'] = data['message'].apply(string_to_list)

    # Apply text preprocessing steps
    data['processed_message'] = data['message'].apply(preprocess_text)
    data['processed_message'] = data['message'].apply(tokenize_message)

    # Convert the message column back to string
    data['processed_message'] = data['message'].apply(list_to_string)

    return data


# Function to plot sentiment label distribution
def plot_sentiment_distribution(data, sentiment_mapping):
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=data, x='sentiment')
    ax.set_xlabel('Sentiment Label')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Label Distribution')
    ax.set_xticklabels([sentiment_mapping[label] for label in ax.get_xticks()], rotation=30, size=10)
    st.pyplot(plt)

# Function to plot model evaluation metrics
def plot_model_metrics(metrics_df):
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    chart = alt.Chart(metrics_df).transform_fold(
        metric_names,
        as_=["Metric", "Value"]
    ).mark_line().encode(
        x="Model",
        y="Value",
        color="Metric",
        tooltip=["Metric", "Value"]
    ).properties(
        width=500,
        height=300
    )

    st.altair_chart(chart)

# Train and evaluate models
def train_and_evaluate_models(model_names, training_data, tweet_cv):
    trained_models = {}
    metrics = []

    for model_name in model_names:
        if model_name == "Logistic Regression":
            model = LogisticRegression()
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'saga']
            }
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 5, 10]
            }
        else:
            raise ValueError("Invalid model name.")

        X_train = tweet_cv.transform(training_data['message']).toarray()
        y_train = training_data['sentiment']

        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
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

# Main function
def main():
    # Streamlit configurations
    st.set_page_config(layout="wide")

    # Load the vectorizer and models
    vectorizer = load_vectorizer()
    loaded_models = load_models()
    trained_model_names = list(loaded_models.keys())  # Get the trained model names


    # Load the raw data
    raw_data = load_raw_data()

    # Sentiment label mapping
    sentiment_mapping = {0: 'Anti', 1: 'Neutral', 2: 'Pro', 3: 'News'}

    st.title("Tweet Classifier")
    st.subheader("Climate change tweet classification")

    # Sidebar options
    options = ["Prediction", "Training", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options, key="selection")

    if selection == "Information":
        st.sidebar.info("General Information")
        st.sidebar.markdown("Some information here")

        show_raw_data = st.sidebar.checkbox('Show Raw Data')

        st.subheader("Sentiment Label Distribution")
        plot_sentiment_distribution(raw_data, sentiment_mapping)

        if show_raw_data:
            st.subheader("Raw Twitter Data and Label")
            st.write(raw_data[['sentiment', 'message']])

    elif selection == "Prediction":
        st.info("Prediction with ML Models")
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            if not tweet_text:
                st.warning("Please enter some text.")
            else:
                try:
                    vect_text = vectorizer.transform([tweet_text]).toarray()

                    # Model selection
                    model_name = st.selectbox("Select Model", list(loaded_models.keys()), key="model_selection")
                    model = loaded_models[model_name]
                    prediction = model.predict(vect_text)
                    sentiment_label = sentiment_mapping[prediction[0]]

                    st.success("Text Categorized as: {}".format(sentiment_label))
                except Exception as e:
                    st.error("An error occurred while performing classification.")
                    st.error(str(e))

    elif selection == "Training":
        st.info("Model Training")
        uploaded_file = st.file_uploader("Upload Training Data (CSV)", type="csv")

        if uploaded_file:
            training_data = pd.read_csv(uploaded_file)
            st.subheader("Training Data Preview")
            st.write(training_data)

            train_model_names = st.multiselect("Select Models for Training", trained_model_names, key="training_model_selection")  # Use trained model names in the multiselect widget

            if st.button("Train Models"):
                try:
                    preprocessed_data = preprocess_data(training_data)
                    trained_models, metrics_df = train_and_evaluate_models(train_model_names, preprocessed_data, vectorizer)
                    save_models(trained_models)  # Save the trained models
                    st.success("Models trained and saved successfully.")

                    st.subheader("Model Evaluation Metrics")
                    st.write(metrics_df)

                    st.subheader("Model Performance Comparison")
                    plot_model_metrics(metrics_df)
                except Exception as e:
                    st.error("An error occurred while training the models.")
                    st.error(str(e))


if __name__ == '__main__':
    main()
