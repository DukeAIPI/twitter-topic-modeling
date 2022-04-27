import numpy as np
import pandas as pd
from scripts.lib_functions import get_tweets, convert_to_df
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from tqdm import tqdm
import os

os.environ["SEARCHTWEETS_ENDPOINT"] = "https://api.twitter.com/2/tweets/search/recent"

#
#
# IMPORTANT!! This file is meant to run locally (it runs on terminal by running >> python non-deep-learning.py)
# In order to run this file, you will need your own twitter API access credentials, since I cannot for security
# purposes give you access to my API creds. You will need to set three environment variables on your local environment
# for this file to run properly. (1) SEARCHTWEETS_BEARER_TOKEN, (2), SEARCHTWEETS_CONSUMER_KEY, (3), SEARCHTWEETS_CONSUMER_SECRET.
#
# **These three access credentials you get directly from twitter developer portal for API v2 access, which is free.**
#

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in tqdm(enumerate(ldamodel[corpus])):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

if __name__ == "__main__":
    term = input("Please input term to search: ")
    topics = input("Please input number of topics to have: ")
    messy_tweets = get_tweets(term, max_tweets=1000) # search term
    tweets = convert_to_df(messy_tweets) # messy tweets
    docs_clean = [clean(doc).split() for doc in tweets["tweet"]] # tokenized, put everything lowercase, removed stop words, removed punctuation, lemmatized everything.
    dictionary = gensim.corpora.Dictionary(docs_clean)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs_clean]
    lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                         num_topics=topics, 
                                         id2word=dictionary, 
                                         passes=4, 
                                         workers=4,
                                         random_state=21)
    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=bow_corpus, texts=tweets["tweet"])
    print(df_topic_sents_keywords)
    print("")
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))