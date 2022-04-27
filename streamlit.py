import streamlit as st
from scripts.lib_functions import get_tweets, convert_to_df, clean_tweets, get_clusters
import os
from bertopic import BERTopic

import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.title("Twitter Topic Modeling")
st.image("./images/Twitter-logo.png")

term = st.text_input("Term you would like to search:")

if term:
    messy_tweets = get_tweets(term, max_tweets=500) # search term
    tweets = convert_to_df(messy_tweets) # messy tweets
    processed_tweets = clean_tweets(tweets) # stop words removed, lemmatized, removed @'s and web links
    
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(processed_tweets["tweet"])
    info = topic_model.get_topic_info()

    #st.dataframe(processed_tweets)
    clusters = get_clusters(info)
    for i in range(clusters.shape[0]):
        st.markdown("## **"+f"Topic {i} (" + str(clusters["Count"].iloc[i]) + " tweets): " + clusters["clean_clusters"].iloc[i][0] + ", " + clusters["clean_clusters"].iloc[i][1] + ", " + clusters["clean_clusters"].iloc[i][2] + ", " + clusters["clean_clusters"].iloc[i][3] + "**")
