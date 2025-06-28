# Comment-Sentiment-Analyzer
Quick-Start Guide


To begin, make sure that you have Python 3 installed, preferably version 3.9 as this is the version that the program was made on.

Run the following command to make sure that you have all the requirements:

pip install flask tensorflow pandas scikit-learn matplotlib seaborn

Next go into the directory and run:

python comment_sentiment_analyzer.py

Click on the link in the console or go to http://127.0.0.1:5000/ to see the web app. 

Type in a comment into the box then click “Predict Sentiment” to get your results


Optional Features

Run python comment_sentiment_analyzer.py -train to train a new model(may take 30 minutes or more)

Run python comment_sentiment_analyzer.py -evaluate to make a new eval_output.json, which can be used with python make_visuals.py to make new graphs in the static directory


