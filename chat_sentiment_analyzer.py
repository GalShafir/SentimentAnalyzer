import pandas as pd
import re
import string
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys
import os

from nltk import FreqDist
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def remove_noise(tweet_tokens, stop_words=()):
    """"
    Some words are generally irrelevant when processing language.
    This function will help us clean the data (links, stop words, punctuation and special characters, etc..)
    """
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_frequency(words):

    freq_dist_pos = FreqDist(words)
    print("Top 3 frequently words in the sentence:")
    print(freq_dist_pos.most_common(3))


def process_message(message, model):

    custom_tokens = remove_noise(word_tokenize(message))
    # get_frequency(custom_tokens)
    sentiment = model.classify(dict([token, True] for token in custom_tokens))
    # print("Sentiment: ", sentiment)

    return sentiment


def create_graph(dataframe):

    # Use white grid plot background from seaborn
    sns.set(font_scale=1.5, style="whitegrid")

    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(10, 10))

    # Add x-axis and y-axis
    ax.plot(dataframe['createdAt'],
            dataframe['sentiment'],
            color='purple')

    # Set title and labels for axes
    ax.set(xlabel="Date",
           ylabel="Sentiment ",
           title="Sentiment Change over time")

    plt.gca().invert_yaxis()
    plt.show()


def main():

    # making dataframe
    #csv_file_name = r"Archive\Global Surface Temperature Change for February_7870804823.csv"

    if len(sys.argv) != 2:
        print("Please enter a csv file wrapped with quotes")
        exit()

    csv_file_name = sys.argv[1]
    df = pd.read_csv(csv_file_name)

    # output the dataframe
    print(df)

    filename = 'sentiment_analyzer_trained_model.sav'

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))

    results = ["1" if process_message(x, loaded_model) == "Positive" else "0" for x in df['raw_message']]
    df['sentiment'] = results

    print(df)
    base = os.path.basename(csv_file_name)
    new_csv_name = str(base.split(".")[0]) + "_with_sentiments.csv"
    df.to_csv(new_csv_name, encoding='utf-8', index=False)
    create_graph(df)


if __name__ == '__main__':
    main()
