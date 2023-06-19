import time
import pandas as pd
import streamlit as st
from preprocess import most_common_word_plot,preprocess_data, train_models, generate_metrics_chart, train_vectorizer, preprocess_text, save_trained_models, model_names, load_model, load_vectorizer, display_vect_metrics



def update_chart():
     # Perform actions based on the updated value of split_ratio
    # Update the chart or perform other operations

    # Example: Update a chart
    # chart = generate_chart(split_ratio=value)
    # st.altair_chart(chart)

    # Example: Recalculate metrics
    # metrics = calculate_metrics(split_ratio=value)
    # st.write("Metrics:", metrics)

    # Example: Trigger other logic
    # do_something(split_ratio=value)

    # Additional logic here if needed
    return
# Main function
def main():
    #st.set_page_config(layout="wide")
    st.set_page_config(page_title="DN3-Classifix", page_icon="⬇", layout="centered")
    sentiment_mapping = {-1: 'Anti', 0: 'Neutral', 1: 'Pro', 2: 'News'}
    st.title("Tweet Classifier")
    st.sidebar.title("Options")
    metrics_df = {}
    trained_models = {}
    model = None
    page = st.sidebar.selectbox(
        "Choose a page", ["Home", "Train Models", "Predict"])

    if page == "Home":
        st.write("Welcome to the Tweet Classifier app!")
        #st.write("Use the sidebar to navigate to different pages.")
        st.info('Use the sidebar to navigate to different pages.')
        a = st.sidebar.radio('Summary:', ["Corpus", "Models"])

        col1, col2 = st.columns(2)
        col1.write("Additional Details will be disaplyed here")

        col2.write("Additional Details will be disaplyed here")
        chart = most_common_word_plot()
        st.altair_chart(chart)
    elif page == "Train Models":
        st.sidebar.write("### Train Models")
        st.sidebar.write("Load and preprocess the training data:")

        # Custom training data upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            raw_data = pd.read_csv(uploaded_file)
            preprocessed_data = preprocess_data(raw_data)
            col1, col2, col3 = st.columns(3)

            with st.sidebar:
                ticker_dx = st.slider("MAX Features", min_value=0, max_value=100000, step=1, value=0)

            if st.sidebar.button("Fit Vectorizer"):
                if ticker_dx>1:
                    with st.spinner("Training Vectorizer..."):
                        time.sleep(2)
                        vectorizer = train_vectorizer(preprocessed_data)
                    st.sidebar.success('Fitting vectorizer complete!')
                    display_vect_metrics(vectorizer, preprocessed_data, st)

            selected_models = st.sidebar.multiselect(
                "Select models to train", model_names)

            with st.sidebar:
                split_ratio = st.slider("Split Ratio", min_value=0.0, max_value=1.0, step=0.1, value=0.1)
                #split_ratio = st.slider("Split Ratio", min_value=0.0, max_value=1.0, step=0.1, value=0.1, label="Select Split Ratio:", key="split_ratio", help="Adjust the split ratio between test and train datasets", format="%.1f")
                k_fold = st.radio('Use K-Fold', ['No', 'Yes'])
                if k_fold == "Yes":
                    num_folds = st.slider("K-Folds", min_value=0, max_value=10, step=1, value=2)

            if st.sidebar.button("Train The Models"):
                vectorizer = load_vectorizer()
                if split_ratio>0.2:
                    with st.spinner("Training models..."):
                        time.sleep(2)
                        trained_models, metrics_df = train_models(
                        selected_models, preprocessed_data, vectorizer, split_ratio)
                    st.sidebar.success("Models trained successfully.")
                    st.table(raw_data[["tweetid","message", "processed_text", "sentiment"]].head())
                    


                    with st.spinner("Saving models..."):
                        time.sleep(2)
                        save_trained_models(trained_models)
                    st.success("Models saved successfully.")

    elif page == "Predict":
        st.write("### Predict")
        message = st.text_input("Enter a tweet:")
   
        selected_model = st.selectbox("Select a model", model_names)

        if st.button("Predict"):
            processed_message = preprocess_text(message)
            vectorizer = load_vectorizer()

            X_test = vectorizer.transform([processed_message]).toarray()
            model = load_model(selected_model)
            print(f"{model}")
            prediction = model.predict(X_test)[0]
            st.write("Predicted Sentiment: ", sentiment_mapping[prediction])

    # Display model evaluation metrics and trained models
    st.subheader("Model Evaluation Metrics")
    if page == "Predict" and model:
        st.write(model)
        bar_chart = most_common_word_plot()
        st.altair_chart(bar_chart)
    if page == "Train Models" and metrics_df is not None:
        try:
            if not metrics_df.empty:
                st.write("Models Metrics")
                st.write(metrics_df)
                generate_metrics_chart(metrics_df)
                
        except Exception as e:
            st.write("No metrics to Display")

        st.subheader("Trained Models")
        if train_models:
            for model_name, _ in trained_models.items():
                st.write(" - ", model_name)

if __name__ == "__main__":
    main()
