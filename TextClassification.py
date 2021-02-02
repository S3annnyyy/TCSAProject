import sys
import json
import re
import time
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
analyser = SentimentIntensityAnalyzer()

# updating VADER sentiment lexicons with words adapted for context from SGExams to improve accuracy
updated_lexicon = {
    "meme": 5.0,
    "F*ck": -5.0,
    "lost": -2.0
}
analyser.lexicon.update(updated_lexicon)

"""
Classify dataset texts into different categories and run sentiment analysis on it and visualise the data
"""


def main():

    # check command line argument
    if len(sys.argv) != 2:
        sys.exit("Usage: python TextClassification.py dataset")

    # time measurement
    start_time = time.time()

    # set random seed
    np.random.seed(50)

    # read dataset using pandas
    corpus = pd.read_csv(r'C:\Users\Sean\PycharmProjects\SentAnalysisproject\{}'.format(sys.argv[1]),
                         encoding="latin-1")

    # data preprocessing
    # removing any blank rows
    corpus.dropna(inplace=True)
    # set all words to lowercase
    corpus["post_text"] = [text.lower() for text in corpus["post_text"] if type(text) == str]
    # tokenize words
    corpus["post_text"] = [word_tokenize(text) for text in corpus["post_text"]]
    # removing stopwords and perform Word Lemmatization
    stop_words = set(stopwords.words('english'))
    for index, text in enumerate(corpus["post_text"]):
        finalized_data = []
        lemmatized_word = WordNetLemmatizer()
        # Word Lemmatization
        for i in range(len(text)):
            word = text[i]
            tag = get_wordnet_posttag(word)
            # eliminating stopwords and filtering out non alphabets
            if word not in stop_words and word.isalpha():
                finalized_word = lemmatized_word.lemmatize(word, tag)
                finalized_data.append(finalized_word)
                # remove float
                finalized_data = remove_float(finalized_data)
        # The final preprocessed words will be stored under "text_final"
        corpus.loc[index, "text_final"] = str(finalized_data)
    # set type to str to avoid error in vectorizer
    corpus["text_final"] = corpus["text_final"].astype(str)

    # getting classification label for corpus["text_final"]
    for index, header in enumerate(corpus["title"]):
        classified_label = []
        temp = get_title_label(header)  # get back a title or None if not present
        if temp is None:  # meaning to say label not properly classified
            classified_label.append("None")
        else:
            classified_label.append(temp)
        # Final processed title stored under label
        corpus.loc[index, "label"] = str(classified_label)
    # code test #2
    # column = ["title", "label"]
    # df1 = pd.DataFrame(corpus, columns=column)
    # print(df1)  # check is correct

    # code test #3
    # for index, label in enumerate(corpus["label"]):
    #     if label is None:
    #         print(label, index)  # check verifies all have specified labels

    # Training ML model
    # Preparing training and testing datasets
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(corpus["text_final"], corpus["label"], test_size=0.3)
    # Encoding labels
    encoder = LabelEncoder()
    Test_Y = encoder.fit_transform(Test_Y)
    Train_Y = encoder.fit_transform(Train_Y)
    # Word Vectorization
    vectorizer = TfidfVectorizer(max_features=10000)
    # learn vocabulary
    vectorizer.fit(corpus["text_final"])
    # turn words to vectors via transform
    Train_X_tfidf = vectorizer.transform(Train_X)
    Test_X_tfidf = vectorizer.transform(Test_X)
    # code test #2
    # print(vectorizer.vocabulary_)
    # print(Train_X_tfidf)

    # Support Vector Machine Algorithm
    # fit training dataset onto classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_tfidf, Train_Y)
    # predict labels
    predictions = SVM.predict(Test_X_tfidf)

    # Evaluating SVM model
    # Accuracy
    print("SVM Accuracy Score: %s" % round(accuracy_score(predictions, Test_Y) * 100, 1))
    # Classification report
    print("Detail")
    print(classification_report(Test_Y, predictions))

    # Sentiment Analysis
    corpus["title"] = corpus["title"].astype(str)
    for index, sentence in enumerate(corpus["title"]):
        sentiment_rating = sentiment_analyzer(sentence)
        corpus.loc[index, "sentiment_ratings"] = float(sentiment_rating)
    # code test #1
    # column = ["title", "sentiment_ratings"]
    # df1 = pd.DataFrame(corpus, columns=column)
    # print(df1)  # code check is good, sentiment analysis is pretty accurate

    # export processed corpus to csv file for visualisation
    corpus.to_csv(r'C:\Users\Sean\PycharmProjects\SentAnalysisproject\wordcloudvis1000.csv',
                  columns=["label", "text_final", "sentiment_ratings", "time"])

    # Prepping data for data visualization
    # categorising labels with their sentiment values
    titles = ["['Uni']", "['JC']", "['Poly']", "['Sec']", "['All']"]
    column = ["label", "sentiment_ratings"]
    df2 = pd.DataFrame(corpus, columns=column)
    df2 = df2.sort_values(by=["label"])
    for i in range(len(df2)):
        label = df2.at[i, 'label']
        for title in titles:
            if label == title:
                classify_sentiment_value(df2.loc[i].at['sentiment_ratings'], label)
    # export database to each label
    for category in database:
        for term in database[category]:
            assign_to_label(term, category)
    # add "All" category to the rest of the categories
    add_all(results)
    # export data to json file
    with open('sentiment_data.json', 'w') as data:
        json.dump(results, data)
    print("--- %s min ---" % round(((time.time() - start_time) / 60), 1))


def get_wordnet_posttag(word):
    """This function takes in a word which then allocates a tag. Eg: "fish" gets a tag v which is VERB"""
    # added in [0][1][0] so as to get main letter. eg VBG, with [0][1][0] gets back V only
    tagged = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wn.ADJ,
        "N": wn.NOUN,
        "V": wn.VERB,
        "R": wn.ADV
    }
    return tag_dict.get(tagged, wn.NOUN)  # The OG description is always NOUN


def get_title_label(sentence):
    """This function takes in a sentence and uses RegEx or Regular Expression to check
    if string contains specified search pattern and returns back label"""
    labels = {
        "Uni": ["Uni"],
        "JC": ["A levels", "A level", "JC", "Junior College"],
        "Poly": ["Poly"],
        "Sec": ["O levels", "O-Levels", "N levels", "N level"],
        "All": ["Jobs", "RANT", "META", "meme"]
    }
    for label in labels:
        for subset_variations in labels[label]:
            if type(sentence) == str:
                term = re.compile(r'\b{}'.format(subset_variations), flags=re.IGNORECASE).search(sentence)
                if term:  # title name contains keyword function is looking for
                    return label
                else:
                    continue
            else:
                return None


def remove_float(sentence):
    pattern = '[0-9]'
    sentence = [re.sub(pattern, '', i) for i in sentence]
    return sentence


def sentiment_analyzer(sentence):
    score = analyser.polarity_scores(sentence)
    return score['compound']


def classify_sentiment_value(value, label):
    if value >= 0.5:
        database[label]["Very Positive"] += 1
    elif 0.05 <= value < 0.5:
        database[label]["Positive"] += 1
    elif -0.05 < value < 0.05:
        database[label]["Neutral"] += 1
    elif -0.05 >= value > -0.5:
        database[label]["Negative"] += 1
    elif value <= 0.5:
        database[label]["Very Negative"] += 1


def assign_to_label(word, education_type):
    if education_type == "['Uni']":
        results["University"].append(database[education_type][word])
    elif education_type == "['JC']":
        results["Junior College"].append(database[education_type][word])
    elif education_type == "['Poly']":
        results["Polytechnic"].append(database[education_type][word])
    elif education_type == "['Sec']":
        results["Secondary School"].append(database[education_type][word])
    else:
        results["All"].append(database[education_type][word])


def add_all(data):
    x = data['All']
    for i in range(len(x)):
        value = x[i]
        for term in data:
            data[term][i] += value
    del data['All']


database = {
        "['Uni']": {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "Very Positive": 0},
        "['JC']": {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "Very Positive": 0},
        "['Poly']": {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "Very Positive": 0},
        "['Sec']": {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "Very Positive": 0},
        "['All']": {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "Very Positive": 0}
    }

results = {
        "University": [],
        "Junior College": [],
        "Polytechnic": [],
        "Secondary School": [],
        "All": []
    }

if __name__ == "__main__":
    main()
