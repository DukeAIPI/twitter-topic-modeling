from scripts.lib_functions import get_tweets, convert_to_df, clean_tweets
import scripts.nltkmodules
import os
from bertopic import BERTopic

if __name__ == "__main__":
    messy_tweets = get_tweets("Elon Musk", max_tweets=400)
    tweets = convert_to_df(messy_tweets)

    processed_tweets = clean_tweets(tweets)

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(processed_tweets["tweet"])
    info = topic_model.get_topic_info()
    print(info)