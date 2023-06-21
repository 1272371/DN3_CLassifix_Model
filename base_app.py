import time
import pandas as pd
import streamlit as st
from preprocess import display_vect_metrics, most_common_word_plot, metrics_df_path, load_raw_data, preprocess_data, train_models, generate_metrics_chart, train_vectorizer, preprocess_text, save_trained_models, model_names, load_model, load_vectorizer

# Define sentiment mapping as a global variable
sentiment_mapping = {-1: 'Anti', 0: 'Neutral', 1: 'Pro', 2: 'News'}

# Function to load and preprocess the training data
def load_and_preprocess_data(uploaded_file):
    raw_data = pd.read_csv(uploaded_file)
    preprocessed_data = preprocess_data(raw_data)
    preprocessed_data["mapped_sentiment"] = preprocessed_data["sentiment"].map(sentiment_mapping)
    return preprocessed_data


# Function to load the specified models
def load_models(model_names=model_names):
    models = {}
    for model_name in model_names:
        model = load_model(model_name)
        if model is not None:
            models[model_name] = model
    return models


# Function to train the selected models
def train_selected_models(selected_models, preprocessed_data, vectorizer, split_ratio):
    trained_models, metrics_df = train_models(selected_models, preprocessed_data, vectorizer, split_ratio)
    return trained_models, metrics_df


# Function to display the trained models
def display_trained_models(models):
    st.subheader("Trained Models")
    if models:
        for model_name, _ in models.items():
            st.write(" - ", model_name)


# Function to display the model evaluation metrics
def display_metrics(metrics_df, display_charts=True):
    st.subheader("Model Evaluation Metrics")
    if not metrics_df.empty:
        st.table(metrics_df)
    if display_charts:
        generate_metrics_chart(metrics_df)


# Main function
def main():
    st.set_page_config(page_title="DN3-Classifix", page_icon="⬇", layout="centered")

    st.title("Tweet Classifier")
    st.sidebar.title("Options")

    sentiment_mapping = {-1: 'Anti', 0: 'Neutral', 1: 'Pro', 2: 'News'}
    metrics_df = load_raw_data(metrics_df_path)

    page = st.sidebar.selectbox(
        "Choose a page", ["Home", "Predict", "Train Models"])

    if page == "Home":
        st.write("Welcome to the Tweet Classifier app!")
        st.info('Use the sidebar to navigate to different pages.')
        summary_type = st.sidebar.radio('Summary:', ["Corpus", "Models"])

        if summary_type == "Corpus":
            bar_chart = most_common_word_plot()
            st.altair_chart(bar_chart)

        elif summary_type == "Models":
            display_trained_models()

    elif page == "Predict":
        st.write("### Predict")
        message = st.text_input("Enter a tweet:")

        selected_model = st.selectbox("Select a model", model_names)

        if st.button("Predict"):
            processed_message = preprocess_text(message)
            vectorizer = load_vectorizer()

            Pred = vectorizer.transform([processed_message]).toarray()
            model = load_model(selected_model)
            prediction = model.predict(Pred)[0]
            st.write("Predicted Sentiment: ", sentiment_mapping[prediction])

    elif page == "Train Models":
        st.info("### Model Training")
        st.sidebar.write("### Train Models")
        st.sidebar.write("Load and preprocess the training data:")

        # Custom training data upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            preprocessed_data = load_and_preprocess_data(uploaded_file)

            with st.sidebar:
                num_features = st.slider("MAX Features", min_value=0, max_value=200000, step=1, value=3000)

            if st.sidebar.button("Fit Vectorizer"):
                with st.spinner("Training Vectorizer..."):
                    time.sleep(2)
                    vectorizer = train_vectorizer(preprocessed_data, num_features)
                st.sidebar.success('Fitting vectorizer complete!')
                display_vect_metrics(vectorizer, preprocessed_data, st)

            selected_models = st.sidebar.multiselect(
                "Select models to train", model_names)

            with st.sidebar:
                split_ratio = st.slider("Split Ratio", min_value=0.0, max_value=1.0, step=0.1, value=0.2)
                k_fold = st.radio('Use K-Fold', ['No', 'Yes'])
                if k_fold == "Yes":
                    num_folds = st.slider("K-Folds", min_value=0, max_value=10, step=1, value=2)

            if st.sidebar.button("Train The Models"):
                vectorizer = load_vectorizer()

                with st.spinner("Training models..."):
                    time.sleep(2)
                    = train_selected_models(
                        selected_models, preprocessed_data, vectorizer, split_ratio)
                st.sidebar.success("Models trained successfully.")

                with st.spinner("Saving models..."):
                    time.sleep(2)
                    save_trained_models(trained_models)
                st.success("Models saved successfully.")

                with st.spinner("Displaying Model Metrics..."):
                    time.sleep(2)
                    display_trained_models(trained_models)
                    display_metrics(metrics_df)

    # Display model evaluation metrics and trained models
    if page == "Home":
        display_metrics(metrics_df)
        models = load_models()
        display_trained_models(models)

    if page == "Predict":
        display_metrics(metrics_df, display_charts=False)
        models = load_models()
        display_trained_models(models)


if __name__ == "__main__":
    main()
