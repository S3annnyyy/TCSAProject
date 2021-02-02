import numpy as np
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
from ast import literal_eval


def main():
    # time measurement
    start_time = time.time()
    # load sentiment dataset from TextClassification for horizontal bar chart
    with open('sentiment_data.json', 'r') as f:
        corpus = json.load(f)
    categories = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    graph_visualisation(corpus, categories)
    plt.title('r/SGExams top 1000 posts sentiment analysis')
    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.3)
    plt.show()

    # process data
    parse_dates = ["time"]
    df = pd.read_csv('wordcloudvis1000.csv', parse_dates=parse_dates)
    # ["label", "text_final", "sentiment_ratings", "time"]
    df.dropna(inplace=True)
    # compare sentiment ratings and label as negative, positive etc
    for index, sentiment_value in enumerate(df["sentiment_ratings"]):
        temp = get_sentiment_label(sentiment_value)
        df.loc[index, "sentiment_label"] = str(temp)
    # code test [Good]
    # column = ["sentiment_ratings", "sentiment_label"]
    # df1 = pd.DataFrame(df, columns=column)
    # print(df1)

    # word cloud plot
    neg_sent = []
    pos_sent = []
    for index, sentence in enumerate(df["text_final"]):
        if df["sentiment_label"][index] == "Negative":
            neg_sent.append(' '.join(map(str, literal_eval(sentence))))
        else:
            pos_sent.append(' '.join(map(str, literal_eval(sentence))))
    pos_text = ' '.join(map(str, pos_sent))
    neg_text = ' '.join(map(str, neg_sent))
    for i in [pos_text, neg_text]:  # positive image pop up first, followed by negative
        wc = WordCloud(max_words=500, background_color='white').generate(i)
        plt.figure()
        plt.axis('off')
        plt.imshow(wc, interpolation='bilinear')
        plt.show()

    # timeline plot
    # sort dates in ascending order
    df.sort_values(by="time", inplace=True, ignore_index=True)
    # convert datetime to string
    # for index, date_time in enumerate(df["time"]):
    #     date = dt.datetime.strftime(date_time.to_pydatetime(), "%#d %b %y")
    #     df.loc[index, "dates"] = date
    # code test [Good]
    # test = df["date"][0]
    # print(test, type(test))

    # classify to specific education level [JC, Uni, Poly, Sec]
    JC = []
    JC_sent = []
    Uni = []
    Uni_sent = []
    Poly = []
    Poly_sent = []
    Sec = []
    Sec_sent = []
    cluster = [JC, Uni, Poly, Sec]
    sent_cluster = [JC_sent, Uni_sent, Poly_sent, Sec_sent]
    for index, date in enumerate(df["time"]):
        if df["label"][index] == "['All']":
            for level in cluster:
                level.append(date)
            for sent_level in sent_cluster:
                sent_level.append(df["sentiment_ratings"][index])
        if df["label"][index] == "['JC']":
            JC.append(date)
            JC_sent.append(df["sentiment_ratings"][index])
        elif df["label"][index] == "['Poly']":
            Poly.append(date)
            Poly_sent.append(df["sentiment_ratings"][index])
        elif df["label"][index] == "['Uni']":
            Uni.append(date)
            Uni_sent.append(df["sentiment_ratings"][index])
        else:
            Sec.append(date)
            Sec_sent.append(df["sentiment_ratings"][index])

    # plotting individual education level plot
    label = ["Junior College", "University", "Polytechnic", "Secondary"]

    for i in range(len(cluster)):
        # rolling mean
        temp_df = pd.DataFrame(list(zip(cluster[i], sent_cluster[i])), columns=["datetime", "sentiment_value"])
        rolling_mean = temp_df.sentiment_value.rolling(20).mean()
        # Plot education levels
        plt.plot(cluster[i], sent_cluster[i], label=label[i], linewidth=0.8)
        plt.plot(cluster[i], rolling_mean, label="Rolling Average", linewidth=1)
        # Setting labels & limits
        # set locator
        locator = mdates.MonthLocator()
        # Format
        fmt = mdates.DateFormatter("%b %y")
        # Set x-axis
        X = plt.gca().xaxis
        X.set_major_locator(locator)
        # Specify formatter
        X.set_major_formatter(fmt)
        # Fine tuning
        # remove borders
        plt.gca().spines["top"].set_alpha(0.0)
        plt.gca().spines["bottom"].set_alpha(0.3)
        plt.gca().spines["right"].set_alpha(0.0)
        plt.gca().spines["left"].set_alpha(0.3)
        plt.legend()
        plt.ylabel("Sentiment")
        plt.xlabel("Month-Year")
        plt.title("{}'s r/SGExams posts Sentiment Analysis".format(label[i]))
        plt.ylim(-1, 1)
        plt.show()
    print("--- %s min ---" % round(((time.time() - start_time) / 60), 1))


def graph_visualisation(results, category_names):
    """This function maps education levels to a list of their responses sentiment of varying degrees"""
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(1, 0),
              loc='upper right', fontsize='small')

    return fig, ax


def get_sentiment_label(value):
    if value > 0.05:
        return "Positive"
    elif -0.05 < value < 0.05:
        return "Neutral"
    elif value < -0.05:
        return "Negative"


if __name__ == "__main__":
    main()
