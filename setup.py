import nltk
import re, string
import random
import pickle

from nltk.tag import pos_tag
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import classify
from nltk import NaiveBayesClassifier



"""
This file should be executed just once, this is necessary for successfully using out main module for sentiment analysis.
"""

# Downloads a sample tweets that will be used to train and test the model.
nltk.download('twitter_samples')

# The punkt module is a pre-trained model that helps you tokenize words and sentences.
# Tokenization is essentially splitting a phrase, sentence, paragraph, or an entire text
# document into smaller units, such as individual words or terms.
# For instance, this model knows that a name may contain a period (like “S. Daityari”)
# and the presence of this period in a sentence does not necessarily end it.
nltk.download('punkt')

# Wordnet is a lexical database for the English language that helps the script determine the base word.
nltk.download('wordnet')

# The averaged_perceptron_tagger resource helps to determine the context of a word in a sentence.
# For example if a word is a noun, verb and etc.
nltk.download('averaged_perceptron_tagger')

# This removes stop words using a built-in set of stop words in NLTK,
# Examples of stop words in English are “a”, “the”, “is”, “are” and etc.
nltk.download('stopwords')
stop_words = stopwords.words('english')


def lemmatize_sentence(tokens):
    """"
    Normalization helps group together words with the same meaning but different forms.
    Without normalization, “ran”, “runs”, and “running” would be treated as different words
    This function will help us normalize our data
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


def remove_noise(tweet_tokens, stop_words = ()):
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


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    """"
    The function is preparing the data to be fed into the model based on Naive Bayes classifier
    """
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def main():
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    # all_pos_words = get_all_words(positive_cleaned_tokens_list)
    # freq_dist_pos = FreqDist(all_pos_words)

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))

    filename = 'sentiment_analyzer_trained_model.sav'
    pickle.dump(classifier, open(filename, 'wb'))


if __name__ == '__main__':
    main()