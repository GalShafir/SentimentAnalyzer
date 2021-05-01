import nltk

"""
This file should be executed just once, this is necessary for successfully using out main module for sentiment analysis.
"""

# Downloads a sample tweets that will be used to train and test the model.
nltk.download('twitter_samples')

# The punkt module is a pre-trained model that helps you tokenize words and sentences.
# Tokenization is essentially splitting a phrase, sentence, paragraph, or an entire text 
# document into smaller units, such as individual words or terms.
# For instance, this model knows that a name may contain a period (like “S. Daityari”) and the presence of this period in a sentence does not necessarily end it.
nltk.download('punkt')

# Wordnet is a lexical database for the English language that helps the script determine the base word.
nltk.download('wordnet')

# The averaged_perceptron_tagger resource helps to determine the context of a word in a sentence.
# For example if a word is a noun, verb and etc.
nltk.download('averaged_perceptron_tagger')

# This removes stop words using a built-in set of stop words in NLTK,
# Examples of stop words in English are “a”, “the”, “is”, “are” and etc.
nltk.download('stopwords')
