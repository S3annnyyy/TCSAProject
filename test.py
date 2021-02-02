import praw
import nltk
import pandas as pd
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

analyser = SentimentIntensityAnalyzer()
reddit = praw.Reddit(client_id='xxxxxxxx',
                     client_secret='xxxxxxxxxxx',
                     user_agent='xxxxxxxxxxx')

# # getting post title
# # test_subreddit = reddit.subreddit("SGExams").hot(limit=10)
# # for post in test_subreddit:
# #     print(post.title)
#
# # getting top 100 posts
# # sgexam = reddit.subreddit("SGExams").top()
# # for post in sgexam:
# #     print(post.title)
#
# getting comments from specific post ( Using top 10 posts )
sgexam = reddit.subreddit("SGExams").top(limit=10)
posts_dict = {
    "title": [],
    "id": [],
    "url": [],
    "comments_num": [],
    "time": []
}
for post in sgexam:
    posts_dict["title"].append(post.title)
    posts_dict["id"].append(post.id)
    posts_dict["url"].append(post.url)
    posts_dict["comments_num"].append(post.num_comments)
    posts_dict["time"].append(datetime.fromtimestamp(post.created))

# txt = posts_dict["time"][0]
# x = re.search(r'^.*\(d{4})-\d{2}-\d{2}.*$', txt).group(1)
# print(x)
# if x:
#     print("success")
# else:
#     print("failure")
# # convert dict into readable format
# data = pd.DataFrame.from_dict(posts_dict, orient='index')
# # print(data)
#
# # test time
# # print(posts_dict["time"][0])
#
# # extract links/id to get comments
# # test with one link first before iteration
# test_url = posts_dict["url"][0]
# # print(test_url)
# # https://www.reddit.com/r/SGExams/comments/icnijz/uni_why_i_disliked_being_in_nus_dentistry_advice/
# # get submission object
# submission = reddit.submission(url=test_url)
# # may run into this AttributeError: 'MoreComments' object has no attribute 'body' if a lot of comments
# # test_comments = []
# # test_replies = []
# # test_titles = []
# # for top_level_comments in submission.comments:
# #     test_comments.append(str(top_level_comments.body))
# #     for second_level_comments in top_level_comments.replies:
# #         test_replies.append(str(second_level_comments.body))
#
# # for title in posts_dict["title"]:
# #     test_titles.append(title)
#
# # test VADER sentiment analysis with 10 comments
# # print(test_comments[0])
#
#
# corpus = pd.read_csv(r'C:\Users\Sean\PycharmProjects\SentAnalysisproject\redditdataset10.csv')
#
# def sentiment_analyzer(sentence):
#     score = analyser.polarity_scores(sentence)
#     print("{:-<65} {}".format(sentence, str(score)))
#     print("Sentiment score:", score['compound'])
#
#
# print(sentiment_analyzer("Neutral"))


# sentiment_analyzer(test_comments[1])
# test is positive, negative sentences represented as negative
# Eg This completely sucks. I hope this gets mainstream media attention and the school is forced to be transparent
# {'neg': 0.242, 'neu': 0.626, 'pos': 0.132, 'compound': -0.3878}
# EDIT: HAHA i realized you have to look at 'compound' instead of neg, neu and pos

# testing sentiment analysis on very long texts
# sentence = corpus["post_text"][0]
# sentiment_analyzer(sentence)
# text is good

# getting subreddit submission title text
# test_title = posts_dict["title"][0]
# test out sentiment analyzer on title
# sentiment_analyzer(test_title)

# testing out 10 submission titles
# for i in range(10):
#     print(test_titles[i])

# testing out submission texts
# sentiment_analyzer(submission.selftext)
# Sentiment score: -0.9994

# testing out is self function to test for text only posts
# link = posts_dict["url"][0]
# submission = reddit.submission(url=link)
# version = submission.is_self
# print(version)
# if it is text gives true, if its not gives false

# testing iteration
# for index, text in enumerate(posts_dict["post_text"]):
#     print(index)
#     print(text)

# testing pol tag and word lemmatization
# word = "swimming"
# tag = pos_tag([word])[0][1][0].upper()
# # gives back [('swimming', 'VBG')] but if you add [0][1][0].upper() will give back V only instead of VBG
# print(tag)

# def get_wordnet_posttag(word):
#     # added in [0][1][0] so as to get main letter. eg VBG, with [0][1][0] gets back V only
#     tagged = pos_tag([word])[0][1][0].upper()
#     tag_dict= {
#         "J": wn.ADJ,
#         "N": wn.NOUN,
#         "V": wn.VERB,
#         "R": wn.ADV
#     }
#     return tag_dict.get(tagged, wn.NOUN)  # The OG description is always NOUN
#
# tag = get_wordnet_posttag("hello")
# print(tag)

# testing out data preprocessing
corpus = pd.read_csv(r'C:\Users\Sean\PycharmProjects\SentAnalysisproject\redditdataset10.csv',
                     encoding="latin-1")


# # removing missing spaces
# corpus["post_text"].dropna(inplace=True)
# # set all words to lowercase
# corpus["post_text"] = [text.lower() for text in corpus["post_text"]]
# # tokenize words
# corpus["post_text"] = [word_tokenize(text) for text in corpus["post_text"]]
#
# for text in corpus["post_text"]:
#     for i in range(len(text)):
#         print(text[i])


# testing Regular expression
sentence1 = "[Uni] Why I Disliked Being in NUS Dentistry (Advice for Future Applicants)"


# def test_function(w):
#     labels = {
#         "Uni": ["Uni"],
#         "JC": ["A levels", "A level", "JC", "Junior College"],
#         "Poly": ["Poly"],
#         "Sec": ["O levels", "O-Levels", "N levels"],
#         "All": ["Jobs", "RANT", "META", "meme"]
#     }
#     for i in labels:
#         for j in labels[i]:
#             test = re.compile(r'\b{}'.format(j), flags=re.IGNORECASE).search(w)
#             if test:
#                 return i
#             else:
#                 continue
#     # if keyword not in labels, test_function will return a None
#
#
# print(test_function(sentence))

# trying to solve typeError
# one = 1
# if type(one) == int:
#     print("success")
# else:
#     print("failure")

# fixing bug
# def remove_float(sentence):
#     pattern = '[0-9]'
#     sentence = [re.sub(pattern, '', i) for i in sentence]
#     return sentence
#
#
# corpus["post_text"] = [word_tokenize(text) for text in corpus["post_text"]]
# corpus["post_text"] = [remove_float(sentence) for sentence in corpus["post_text"]]
#
# print(len(corpus["post_text"][0]))

# labels = {
#         "Uni": ["Uni"],
#         "JC": ["A levels", "A level", "JC", "Junior College"],
#         "Poly": ["Poly"],
#         "Sec": ["O levels", "O-Levels", "N levels"],
#         "All": ["Jobs", "RANT", "META", "meme"]
#     }

# testing data prep methods
# def addition(value):
#     if value == 1:
#         database["['Uni']"]["Negative"] = database["['Uni']"]["Negative"] + 1
#
#
# database = {
#         "['Uni']": {"Negative": 10, "Neutral": 20, "Positive": 30},
#         "['JC']": {"Negative": 0, "Neutral": 0, "Positive": 0},
#         "['Poly']": {"Negative": 0, "Neutral": 0, "Positive": 0},
#         "['Sec']": {"Negative": 0, "Neutral": 0, "Positive": 0},
#         "['All']": {"Negative": 0, "Neutral": 0, "Positive": 0}
#     }
# # University  =[]
# # for categories in database["['Uni']"]:
# #     University.append(database["['Uni']"][categories])
# #
# # print(University)
# results = {
#         "University": [],
#         "Junior College": [],
#         "Polytechnic": [],
#         "Secondary School": []
#     }
# for category in results:
#     print(category)

# final = {
#     'University': [12, 22, 85, 23, 6],
#     'Junior College': [18, 57, 174, 40, 14],
#     'Polytechnic': [3, 7, 24, 7, 3],
#     'Secondary Schx = len(final["All"])
# # value = x[0]
# # final['University'][0] += value
# # print(final['Uniool': [16, 28, 133, 31, 25],
#     'All': [51, 89, 32, 17, 28]
# }
# versity'])
# print(x)

# test to convert pandas time obj labelled str to datetime.datetime for conversion
# test = df["time"][0]  # 2021-01-14 13:44:37
    # temp = test.to_pydatetime(test)
    # product = dt.datetime.strftime(temp, "%#d %b %y")
    # print(product)

# convert dates to D-M_Y
    # for level in cluster:
    #     level.sort(key=lambda date: dt.datetime.strptime(date, '%d %b %y'))

# # rolling mean
    # test = pd.DataFrame(list(zip(JC, JC_sent)), columns=["datetime", "sentiment_value"])
    # rolling_mean = test.sentiment_value.rolling(20).mean()
    # # Set Locator
    # locator = mdates.MonthLocator()
    # # Format
    # fmt = mdates.DateFormatter("%b %y")
    # # Plot education level plots
    # plt.plot(JC, JC_sent, label="JC", linewidth=0.5, color="lightsteelblue")
    # plt.plot(JC, rolling_mean, label="Rolling average", linewidth=1, color="black")
    # # plt.plot(Uni, Uni_sent, label="Uni", linewidth=0.8, color="aquamarine")
    # # plt.plot(Sec, Sec_sent, label="Secondary", linewidth=0.8, color="thistle")
    # # plt.plot(Poly, Poly_sent, label="Poly", linewidth=0.8, color="oldlace")
    #
    # # Set x-axis
    # X = plt.gca().xaxis
    # X.set_major_locator(locator)
    # # Specify formatter
    # X.set_major_formatter(fmt)
    #
    # # Fine tuning
    # # remove borders
    # plt.gca().spines["top"].set_alpha(0.0)
    # plt.gca().spines["bottom"].set_alpha(0.3)
    # plt.gca().spines["right"].set_alpha(0.0)
    # plt.gca().spines["left"].set_alpha(0.3)
    # # Setting labels & limits
    # plt.legend()
    # plt.ylabel("Sentiment")
    # # plt.title("r/SGExams top 1000 posts sentiment analysis")
    # plt.ylim(-1, 1)
    # plt.show()