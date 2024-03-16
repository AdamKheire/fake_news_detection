
# Importing required libraries
import pickle
import streamlit as st
from sklearn.linear_model import LogisticRegression

# Load the logistic regression model

pickle_in_model = pickle.load(open('model1.pkl', 'rb'))
logistic_regression_model = pickle.load(pickle_in_model)

# Load the TF-IDF vectorizer
pickle_in_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
tfidf_vectorizer = pickle.load(pickle_in_vectorizer)

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
