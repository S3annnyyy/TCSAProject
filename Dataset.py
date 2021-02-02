import praw
import time
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

start_time = time.time()
analyzer = SentimentIntensityAnalyzer()
reddit = praw.Reddit(client_id='xxxxx',
                     client_secret='xxxxx',
                     user_agent='xxxxx')

# getting title post and its text from subreddit
# test with 100 first, official use 1000
sgexam = reddit.subreddit("SGExams").top(time_filter='year', limit=500)
posts_dict = {
    "title": [],
    "post_text": [],
    "url": [],
    "text_type": [],
    "time": []
}
for post in sgexam:
    posts_dict["title"].append(post.title if post.is_self is True else "meme post")
    posts_dict["url"].append(post.url)
    posts_dict["text_type"].append(post.is_self)
    posts_dict["time"].append(datetime.fromtimestamp(post.created))
    if type(post.selftext) == str:
        posts_dict["post_text"].append(post.selftext if post.is_self is True else "meme")
    elif type(post.selftext) == float:
        posts_dict["post_text"].append("Neutral")
# convert dict into readable format
data = pd.DataFrame.from_dict(posts_dict, orient='columns')
print("--- %s seconds ---" % round((time.time() - start_time)))

# export dataframe into csv file
data.to_csv(r'C:\Users\Sean\PycharmProjects\SentAnalysisproject\redditdataset500.csv')

