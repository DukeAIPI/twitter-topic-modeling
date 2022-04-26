import pandas as pd
from searchtweets import load_credentials, gen_request_parameters, collect_results
from nltk.corpus import stopwords
import string 
from nltk.stem import WordNetLemmatizer

def get_tweets(term, max_tweets=100):
    credentials = load_credentials(env_overwrite=True)
    query = gen_request_parameters(f"{term} -is:retweet -is:reply lang:en", results_per_call=100, granularity=None)
    tweets = collect_results(query, max_tweets = max_tweets, result_stream_args=credentials)
    return tweets

def convert_to_df(messy_tweets):
    '''Turns messy json repsonse from get_tweets into a clean df.
    
    Args:
        messy_tweets (list of dicts extracted from json response): 'data' part of .json() parse
        
    Returns:
        df: df that has the extracted tweets
    '''
    clean = []
    
    for i in range(len(messy_tweets)):
        for j in range(len(messy_tweets[i]['data'])):
            clean.append(messy_tweets[i]['data'][j]['text'])
    
    df = pd.DataFrame(clean, columns = ['tweet'])
    df = df.drop_duplicates(ignore_index = True)
    return df

def clean_tweets(tweets):
    clean = tweets.copy()
    stop_words = set(stopwords.words('english'))
    punctuations = string.punctuation
    wordnet_lemmatizer = WordNetLemmatizer()
    clean["tweet"] = clean["tweet"].apply(lambda x: ' '.join(wordnet_lemmatizer.lemmatize(word).lower().strip() for word in x.split() if word not in stop_words and word[0]!="@" and word[:5]!="https"))
    return clean