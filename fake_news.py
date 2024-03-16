# Importing required libraries
import pickle
import streamlit as st

# Load the logistic regression model
with open('model1.pkl', 'rb') as file:
    logistic_regression_model = pickle.load(file)

# Load the TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# This is the main function where we define our app
def main():
    # Header of the page
    html_temp = """
    <div style ="background-color:orange;padding:13px">
    <h1 style ="color:white;text-align:center;">Fake News Prediction</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create a text area where the user can enter news information
    news_text = st.text_area("Additional Information", "Type here...")

    result =""
    # When 'Check' is clicked, make the prediction and display the result
    if st.button("Check"):
        result = prediction(news_text)
        st.success('Your predicted news is {}'.format(result))

# Function to make the prediction using the input data
def prediction(news_text):
    if not news_text.strip():
        return "Please enter some text."

    # Transform the input text into numerical features using the TF-IDF vectorizer
    news_text_tfidf = tfidf_vectorizer.transform([news_text])

    # Make prediction using the logistic regression model
    prediction = logistic_regression_model.predict(news_text_tfidf)

    if prediction == 0:
        pred = 'Fake'
    else:
        pred = 'Real'

    return pred

if __name__=='__main__':
    main()
