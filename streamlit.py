import streamlit as st
from scripts.lib_functions import get_tweets, convert_to_df, clean_tweets, get_clusters
import pandas as pd
from bertopic import BERTopic

import nltk

@st.cache(allow_output_mutation=True)
def load_models():
    '''Downloads NLTK packages needed for NLP preprocessing steps, as well as BERTopic model.'''
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    topic_model = BERTopic() # BERTopic model to return
    
    return topic_model

topic_model = load_models()

# Streamlit (frontend) code begins here:
st.title("Twitter Topic Modeling Web Microservice")
st.header("Using machine learning to gain insight from Twitter data.")
st.subheader("Project by Leo Corelli")
st.image("./images/Twitter-logo.png")

st.write("Let's find out what people are actually saying about a given topic! This microservice is going to pull the 1500 most recent tweets on a topic of your choice and then use machine learning to cluster them and present back to you the relevant subtopics, as well as representative tweets for each of those subtopics.")

term = st.text_input("Term you would like to search (please allow ~45 seconds to run):", placeholder="Ex: 'Economy' or 'JetBlue'")

if term:
    messy_tweets = get_tweets(term, max_tweets=1500) # search term
    tweets = convert_to_df(messy_tweets) # messy tweets
    processed_tweets, tweets = clean_tweets(tweets) # stop words removed, lemmatized, removed @'s and web links
    
    try:
        topics, probs = topic_model.fit_transform(processed_tweets["tweet"]) # do topic modeling on the tweets
        info = topic_model.get_topic_info()

        clusters = get_clusters(info)

        rep_docs = topic_model.get_representative_docs()

        pairwise_df = pd.concat([tweets["tweet"], processed_tweets["tweet"]],axis=1)
        pairwise_df.columns = ["original_tweet","cleaned_tweet"]

        pairwise_df["original_tweet"] = pairwise_df["original_tweet"].replace("\n",' ', regex=True)

        for i in range(clusters.shape[0]): # this for loop iterates through variable length cluster results and neatly displays them using the streamlit framework
            st.markdown("## **"+f"Topic {i+1} (" + str(clusters["Count"].iloc[i]) + " tweets): " + clusters["clean_clusters"].iloc[i][0] + ", " + clusters["clean_clusters"].iloc[i][1] + ", " + clusters["clean_clusters"].iloc[i][2] + ", " + clusters["clean_clusters"].iloc[i][3] + "**")
            for j in range(len(rep_docs[i])):
                st.markdown("- " + pairwise_df[pairwise_df["cleaned_tweet"]==rep_docs[i][j]]["original_tweet"].iloc[0])

    except ValueError:
        st.subheader("Not enough tweets on this topic! Please try a more popular topic with at least 1500 related tweets in the last 7 days :)")