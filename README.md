# SentimentAnalyzer

### This project will allow you to analyze in graphical way a chat group sentiments.
#### The projects contain two main scipts:

1. A simple script that gets as an argument a message (simple string) and will return some metrics about the sentence.
   Metrics like: words count, positivity level, etc.
   
2. A script that will get as an input a full group chat table sorted by time. the script will iterate over the table and will represent a graph of the sentiments results (positive or negative). The script will also save a new csv file with a sentiment result next to each line.


##### How to run the first script (simple_sentence_analyzer.py)

```python simple_sentence_analyzer.py "Hello world!"```

The follwoing will get you the results:

```
Top 3 frequently words in the sentence:
[('hello', 1), ('world', 1)]
Sentiments:  Positive
```

##### How to run the second script (chat_sentiment_analyzer.py)

``` python chat_sentiment_analyzer.py "path/to/mycsv_file.csv" ```

The follwoing will get you the result (only for example):

![image](https://user-images.githubusercontent.com/45572842/116785229-155ff980-aaa1-11eb-917b-962f49cd2052.png)


##### What is setup.py

This file is a model training script.
The file is no necessary for users unless they want to improve the the model (sentiment_analyzer_trained_model.sav)
In any other case the model is already ready for any use.

The model is based on Shaumik Daityari article in digitalocean.

##### requirements

In order to use the script some libraries must be installed.
Use the following command to download the libraries:

```pip install -r requirements.txt```

