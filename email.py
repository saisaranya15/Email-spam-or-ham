import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv")

# Streamlit UI
st.title("Email Spam or Ham Classification")

# Data preprocessing
X = df["Message"]
y = df["Category"]

bow = CountVectorizer(stop_words="english")
final_x = pd.DataFrame(bow.fit_transform(X).toarray(), columns=bow.get_feature_names_out())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(final_x, y, test_size=0.25, random_state=23)

# Model training
nav_bay = MultinomialNB()
nav_bay.fit(X_train, y_train)

# Accuracy score button
if st.button("Predict Accuracy"):
    y_pred = nav_bay.predict(X_test)
    st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# User input for prediction
email_input = st.text_input("Enter Email:")

def predict_email(email):
    data = bow.transform([email]).toarray()
    prediction = nav_bay.predict(data)[0]
    st.write(f"Prediction: {prediction}")

if st.button("Predict"):
    predict_email(email_input)
