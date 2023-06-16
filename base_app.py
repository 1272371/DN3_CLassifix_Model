import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import *


# Main function
def main():
    st.set_page_config(layout="wide")
    loaded_models = load_models()
    raw_data = load_raw_data()

    vectorizer = TfidfVectorizer(max_features=100)

    st.title("Tweet Classifier")
    st.sidebar.title("Options")

    page = st.sidebar.selectbox("Choose a page", ["Home", "Train Models", "Predict"])

    if page == "Home":
        st.write("Welcome to the Tweet Classifier app!")
        st.write("Use the sidebar to navigate to different pages.")

    elif page == "Train Models":
        st.write("### Train Models")
        st.write("Load and preprocess the training data:")
        preprocessed_data = preprocess_data(raw_data)

        model_names = st.multiselect("Select models to train", list(loaded_models.keys()))

        if st.button("Train Models"):
            # Show loading slide
            with st.spinner("Training models in progress..."):
                import time
                time.sleep(3)  # Simulating training process (remove this line in your actual code)
                trained_models, metrics_df = train_and_evaluate_models(model_names, preprocessed_data, vectorizer)
            # Hide loading slide and display success message
            st.success("Models trained successfully.")

            save_models(trained_models)  # Save the trained models
            st.success("Models saved successfully.")

            st.subheader("Model Evaluation Metrics")
            st.write(metrics_df)

            for model_name, model in trained_models.items():
                st.write("- ", model_name)

            st.write("Model Evaluation Metrics:")
            st.write(metrics_df)

    elif page == "Predict":
        st.write("### Predict")
        message = st.text_input("Enter a tweet:")
        selected_model = st.selectbox("Select a model", list(loaded_models.keys()))

        if st.button("Predict"):
            processed_message = preprocess_data(pd.DataFrame({'message': [message]}))
            X_test = vectorizer.transform(processed_message['processed_text']).toarray()
            model = loaded_models[selected_model]
            prediction = model.predict(X_test)[0]
            st.write("Predicted Sentiment:", prediction)

if __name__ == "__main__":
    main()
