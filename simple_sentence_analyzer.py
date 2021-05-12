import re
import string
import pickle
import sys

from nltk import FreqDist
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def remove_noise(tweet_tokens, stop_words=None):
    """"
    Some words are generally irrelevant when processing language.
    This function will help us clean the data (links, stop words, punctuation and special characters, etc..)
    """
    if not stop_words:
        stop_words = list()
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

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


def main():
    if len(sys.argv) != 2:
        print("Please enter a sentence wrapped with quotes")
        exit()

    custom_tweet = sys.argv[1]

    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    get_frequency(custom_tokens)

    filename = 'sentiment_analyzer_trained_model.sav'

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))

    print("Sentiments: ", loaded_model.classify(dict([token, True] for token in custom_tokens)))


if __name__ == '__main__':
    main()
