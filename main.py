import yfinance as yf
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Downloading Tesla's stock price data
ticker = "TSLA"
start_date = "2019-03-16"
end_date = "2022-03-16"
tsla = yf.download(ticker, start=start_date, end=end_date)
print(tsla)
# Reading Elon Musk's tweets data
elon_tweets = pd.read_csv("elonmusk_tweets.csv")
elon_tweets = elon_tweets[['date', 'tweet']]
elon_tweets['date'] = pd.to_datetime(elon_tweets['date'])

# Performing sentiment analysis on Elon Musk's tweets
analyzer = SentimentIntensityAnalyzer()
sentiments = []
for tweet in elon_tweets['tweet']:
    scores = analyzer.polarity_scores(tweet)
    sentiment = scores['compound']
    sentiments.append(sentiment)

elon_tweets['sentiment'] = sentiments

# Joining the sentiment scores with the stock price data
merged_df = tsla.merge(elon_tweets, how='left', left_index=True, right_on='date').dropna()

# Creating a new column for the stock price movement
merged_df['price_movement'] = merged_df['Close'].diff().shift(-1).apply(lambda x: 1 if x > 0 else 0)

# Training a machine learning model to predict the stock price movement
X = merged_df[['sentiment', 'Close', 'Volume']]
y = merged_df['price_movement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# API endpoint to predict stock price movement based on a new tweet
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    new_tweet = data['tweet']
    scores = analyzer.polarity_scores(new_tweet)
    sentiment = scores['compound']
    last_close = merged_df['Close'].iloc[-1]
    last_volume = merged_df['Volume'].iloc[-1]
    X_new = pd.DataFrame([[sentiment, last_close, last_volume]], columns=['sentiment', 'Close', 'Volume'])
    predicted_prob = clf.predict_proba(X_new)[0][1]
    if sentiment > 0:
        price_change = last_close * predicted_prob
        response = {
            'prediction': 'The stock price is predicted to go up.',
            'price_change': f'The predicted price change is: {price_change:.2f}'
        }
    elif sentiment == 0:
        price_change = last_close * predicted_prob
        response = {
            'prediction': 'The stock price is predicted to not change.',
            'price_change': f'The predicted price change is: {price_change:.2f}'
        }
    else:
        price_change = last_close * -predicted_prob
        response = {
            'prediction': 'The stock price is predicted to go down.',
            'price_change': f'The predicted price change is: {price_change:.2f}'
        }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
