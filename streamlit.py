import streamlit as st
from scripts.lib_functions import get_tweets, convert_to_df, clean_tweets
import scripts.nltkmodules
import os
from bertopic import BERTopic

st.title("Twitter Topic Modeling")
st.image("./images/Twitter-logo.png")

term = st.text_input("Term you would like to search")

if term:
    messy_tweets = get_tweets(term, max_tweets=500) # search term
    tweets = convert_to_df(messy_tweets) # messy tweets
    processed_tweets = clean_tweets(tweets) # stop words removed, lemmatized, removed @'s and web links

    # st.dataframe(processed_tweets)

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(processed_tweets["tweet"])
    info = topic_model.get_topic_info()

    st.dataframe(info)