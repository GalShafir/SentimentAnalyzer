# SentimentAnalyzer

### This project allows you to analyze a chat group sentiments in a graphical way.
#### The projects contain two main scripts:

1. **`simple_sentence_analyzer`:** Gets a message  as an argument (string) and prints some metrics about the sentence, like: words count, its sentiment.
   
2. **`chat_sentiment_analyzer`:** Gets a full group chat table sorted by time as an input,
   iterates over the table and represents a graph of the sentiments results (positive or negative). 
   The script also saves a new CSV file with a sentiment result next to each line.

### Requirements
- Python version 3.5, 3.6, 3.7, 3.8, or 3.9.
- Install the pip libs from `requirements.txt` file (`pip install -r requirements.txt`).

### How to run the scripts?
- *simple_sentence_analyzer.py:*

   ```python simple_sentence_analyzer.py "Hello world!"```

   **Expected output for example:**
   ```
   Top 3 frequently words in the sentence:
   [('hello', 1), ('world', 1)]
   Sentiments:  Positive
   ```

- *chat_sentiment_analyzer.py:*
  
   ``` python chat_sentiment_analyzer.py "path/to/mycsv_file.csv"```
  
   **Expected output for example:**

   ![image](https://user-images.githubusercontent.com/45572842/116785229-155ff980-aaa1-11eb-917b-962f49cd2052.png)


##### What is setup.py

This file is a model training script.
The file is no necessary for users unless they want to improve the the model (sentiment_analyzer_trained_model.sav)
In any other case the model is already ready for any use.

The model is based on Shaumik Daityari article from digitalocean.
