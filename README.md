# musk-sentiment-analysis
 Stock Price Prediction using Sentiment Analysis
# Stock Price Movement Prediction API

This is a Flask-based API that predicts the movement of Tesla's stock price based on Elon Musk's tweets using sentiment analysis and a machine learning model. The API endpoint `/predict` accepts a new tweet and returns the predicted stock price movement and the expected price change.

## Requirements

- Python 3.6 or higher
- Flask
- Flask-CORS
- NLTK
- scikit-learn
- pandas
- yfinance

## Installation

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages using the following command: `pip install -r requirements.txt`.

## Usage

1. Run the Flask application by executing the following command in your terminal: `python app.py`.
2. Send a POST request to the `/predict` endpoint with a JSON object containing the new tweet: 

{
"tweet": "New tweet by Elon Musk"
}


3. The API will return a JSON object with the predicted stock price movement and the expected price change:

{
"prediction": "The stock price is predicted to go up.",
"price_change": "The predicted price change is: 10.00"
}


## How it works

1. Download Tesla's stock price data from Yahoo Finance using the `yfinance` package.
2. Read Elon Musk's tweets data from a CSV file.
3. Perform sentiment analysis on Elon Musk's tweets using the `SentimentIntensityAnalyzer` from the `NLTK` package.
4. Join the sentiment scores with the stock price data.
5. Create a new column for the stock price movement.
6. Train a machine learning model using the `RandomForestClassifier` from the `scikit-learn` package.
7. Create a Flask application with an API endpoint to predict stock price movement based on a new tweet.
8. When a new tweet is received, perform sentiment analysis on it and use the machine learning model to predict the stock price movement and the expected price change.
9. Return the prediction and the expected price change as a JSON object.

## Credits

This project was inspired by [this article](https://towardsdatascience.com/predicting-stock-price-with-elon-musks-tweets-and-scikit-learn-706b1c080983) by [Siddhant Sadangi](https://towardsdatascience.com/@sidsadangi).

The stock price data was obtained from Yahoo Finance using the `yfinance` package.

The tweets data was obtained using the [Twint](https://github.com/twintproject/twint) package.

The sentiment analysis was performed using the `SentimentIntensityAnalyzer` from the `NLTK` package.

The machine learning model was trained using the `RandomForestClassifier` from the `scikit-learn` package.
