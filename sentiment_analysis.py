# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, accuracy_score
# import re
# from nltk.corpus import stopwords
# import numpy as np
#
# # --- STEP 1: Load Large Dataset ---
# # You must download the CSV file from Kaggle first (e.g., IMDB 50K Movie Reviews)
# FILE_PATH = 'C:/Users/kazi/Downloads/IMDB Dataset.csv/IMDB Dataset.csv'  # <-- UPDATE THIS PATH if needed
# TEXT_COLUMN = 'review'
# SENTIMENT_COLUMN = 'sentiment'
#
# try:
#     # Loading the large dataset
#     data = pd.read_csv(FILE_PATH, encoding='utf-8')
#
#     # 1. Select the relevant columns
#     data = data[[TEXT_COLUMN, SENTIMENT_COLUMN]].copy()
#
#     # 2. Rename columns for consistency (optional but good practice)
#     data.columns = ['text', 'sentiment']
#
#     # 3. Convert labels from text ('positive', 'negative') to numerical (1, 0)
#     data['sentiment'] = data['sentiment'].replace({'positive': 1, 'negative': 0})
#
#     # Optional: Limit dataset size for initial testing on mid-range hardware
#     # data = data.sample(n=50000, random_state=42).copy() # Use a 50K subset if needed
#
#     print(f"Dataset loaded with shape: {data.shape}")
#     print(f"Class distribution:\n{data['sentiment'].value_counts()}")
#
# except FileNotFoundError:
#     print(
#         f"Error: The file '{FILE_PATH}' was not found. Please download the Kaggle dataset and ensure the path is correct.")
#     # Exit or handle error gracefully
#     exit()
#
# data['text'] = data['text'].astype(str)
#
#
# # --- 2. Text Preprocessing ---
# def clean_text(text):
#     # This dataset contains HTML tags like <br />, so we remove them
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'http\S+|www\S+|@\S+', '', text)
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     return text.lower()
#
#
# data['text_clean'] = data['text'].apply(clean_text)
# X = data['text_clean']
# y = data['sentiment']
#
# # --- 3. Split Data with Stratification ---
# # Stratify=y ensures balanced class representation in train and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )
#
# # --- 4. Feature Extraction (TF-IDF Vectorization) ---
# stop_words = stopwords.words('english')
# # Using a larger max_features is common for large datasets
# vectorizer = TfidfVectorizer(
#     stop_words=stop_words,
#     max_features=10000,
#     ngram_range=(1, 2)
# )
#
# print("\nFitting and transforming data...")
# # This step can take a few minutes on a large dataset
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)
#
# print(f"Training data (Vectorized) shape: {X_train_vec.shape}")
# print(f"Testing data (Vectorized) shape: {X_test_vec.shape}")
#
# # --- 5. Model Training (SVM) ---
# print("\nTraining SVM Model...")
#
# # class_weight='balanced' helps mitigate the effect of any slight class imbalance
# # NOTE: Training on a large dataset will be TIME-CONSUMING (can take tens of minutes or more)
# # due to the complexity of the SVM algorithm on high-dimensional data.
# svm_model = SVC(
#     kernel='linear',
#     C=1.0,
#     random_state=42,
#     class_weight='balanced',
#     # Use LinearSVC for very large datasets if SVC is too slow
#     # from sklearn.svm import LinearSVC
#     # svm_model = LinearSVC(C=1.0, random_state=42)
# )
#
# svm_model.fit(X_train_vec, y_train)
# print("Training complete.")
#
# # --- 6. Prediction and Evaluation ---
# y_pred = svm_model.predict(X_test_vec)
# accuracy = accuracy_score(y_test, y_pred)
#
# print("\n## Evaluation Results 📊")
# print(f"Accuracy: {accuracy:.4f}")
# print("\nClassification Report:")
# print(classification_report(
#     y_test,
#     y_pred,
#     target_names=['Negative (0)', 'Positive (1)'],
#     zero_division=0  # Prevents the UndefinedMetricWarning
# ))
#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import re
from nltk.corpus import stopwords
import numpy as np
import streamlit as st
import nltk

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


# --- Utility Functions (Preprocessing) ---
@st.cache_data
def clean_text(text):
    """Applies necessary cleaning to text for consistent processing."""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)  # Remove URLs and Mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation/numbers
    return text.lower()


# --- Core ML Pipeline (Wrapped in a Caching Function) ---
# @st.cache_resource ensures the heavy operations (loading data, training model)
# are only run once, making the app fast after the first run.
@st.cache_resource
def train_sentiment_model(file_path):
    """Loads data, trains the SVM model, and returns the model and vectorizer."""

    st.info(f"Loading and training model on data from: {file_path}. This may take a few minutes...")

    # Define columns based on your Kaggle data
    TEXT_COLUMN = 'review'
    SENTIMENT_COLUMN = 'sentiment'

    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        data = data[[TEXT_COLUMN, SENTIMENT_COLUMN]].copy()
        data.columns = ['text', 'sentiment']
        data['sentiment'] = data['sentiment'].replace({'positive': 1, 'negative': 0})
        data['text'] = data['text'].astype(str)

    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please check the path.")
        return None, None, None

    # Preprocessing
    data['text_clean'] = data['text'].apply(clean_text)
    X = data['text_clean']
    y = data['sentiment']

    # Splitting data (using stratification for robustness)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Extraction (TF-IDF)
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=10000,
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    # Model Training (SVC with linear kernel and class weights)
    svm_model = SVC(
        kernel='linear', C=1.0, random_state=42, class_weight='balanced'
    )
    svm_model.fit(X_train_vec, y_train)

    # Optional: Display evaluation metrics (can be removed for production demo)
    # X_test_vec = vectorizer.transform(X_test)
    # y_pred = svm_model.predict(X_test_vec)
    # st.success(f"Model Training Complete! Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return svm_model, vectorizer


# --- Streamlit UI Setup ---

st.set_page_config(page_title="SVM Sentiment Analyzer", layout="wide")

st.title("🎬 Movie Review Sentiment Analyzer (SVM)")
st.markdown("---")

# **Define the path to your dataset here!**
FILE_PATH = 'C:/Users/kazi/Downloads/IMDB Dataset.csv/IMDB Dataset.csv'

# Load the model and vectorizer
model, vectorizer = train_sentiment_model(FILE_PATH)

if model and vectorizer:

    st.header("1. Input a Review")

    # Text area for user input
    user_input = st.text_area(
        "Paste a movie review here:",
        height=150,
        placeholder="e.g., This movie was absolutely phenomenal and the acting was superb. A must-watch!",
        key="review_input"
    )

    # Button to trigger prediction
    if st.button("Analyze Sentiment", type="primary") and user_input:

        st.header("2. Prediction Result")

        with st.spinner('Analyzing...'):

            # 1. Preprocess the input text
            cleaned_input = clean_text(user_input)

            # 2. Vectorize the cleaned text (using .transform())
            X_input_vec = vectorizer.transform([cleaned_input])

            # 3. Predict the sentiment
            prediction = model.predict(X_input_vec)[0]

            # 4. Determine the result label and emoji
            if prediction == 1:
                result_text = "Positive Sentiment 👍"
                color = "green"
            else:
                result_text = "Negative Sentiment 👎"
                color = "red"

            # Display the result
            st.markdown(
                f"<h3 style='color: {color}; text-align: center;'>{result_text}</h3>",
                unsafe_allow_html=True
            )

            # Optional: Show prediction confidence (if using a model that provides it, like probability, which SVC with kernel='linear' doesn't easily do)
            # st.caption("Note: SVC provides a direct classification, not a probability score.")

    st.markdown("---")
    st.markdown("### Model Details")
    st.write(f"**Model:** Support Vector Classifier (SVC, linear kernel)")
    st.write(f"**Vectorizer:** TF-IDF (max features: 10,000)")