# Miners-of-Wallstreet
B565 Final Project
Abstract
The data mining problem at hand is to predict the randomness of the stock market using machine learning algorithms to identify complex patterns in large amounts of data that may not be immediately apparent to human analysts. By analyzing past market trends, news articles, and other factors, machine learning models can make predictions about future market movements with a high degree of accuracy. This can be valuable for investors and traders who need to make informed decisions about buying and selling stocks. The problem is important as data mining can help identify hidden patterns and uncover significant trends in large amounts of financial data, develop predictive models that can forecast upcoming market trends and its effects on stock prices, allowing investors to make more informed decisions about where/when/how to invest their money efficiently. Obtaining the necessary data involves scraping historical stock prices, news headlines, and sentiment data from various sources such as Yahoo Finance, Alpha Vantage, Quandl, and FINVIZ for tickers filtered based on indices from SP 500, Dow Jones, and some Nasdaq indices, using libraries like BeautifulSoup and Scrapy. The data features include market cap, P/E ratio, revenue, profit margin, volume, and others. Clustering techniques such as K-means are used to group similar companies together based on their financial data.



Data Collection and EDA

There are thousands of tickers/companies in the market, to select the top n companies, we have used k means clustering as the initial approach in gathering the data.

For the initial step of identifying the companies of data, have scraped the data from finviz and stored the fundamental benchmarking indices with stock names in a data frame. Beautiful Soup library is used to perform the analysis.

We filtered the data based on indices from S&P 500, Dow Jones and some Nasdaq indices.

S&P 500 - The S&P 500 is a stock market index that measures the stock performance of 500 large companies listed on US stock exchanges. 
These companies are selected based on market capitalization, liquidity, and other factors. Some of the well-known companies in the S&P 500 index include Apple, Microsoft, Amazon, Facebook, and Tesla.
DJIA - The Dow Jones Industrial Average (DJIA) is a stock market index that measures the stock performance of 30 large companies listed on US stock exchanges.
These companies are selected based on their reputation, size, and sector representation. Some of the well-known companies in the DJIA include Boeing, Coca-Cola, Goldman Sachs, and McDonald's.
NASDAQ-
The NASDAQ Composite is a stock market index that measures the stock performance of more than 3,000 companies listed on the NASDAQ stock exchange. 
These companies are typically technology-based and include well-known companies like Apple, Amazon, Microsoft, and Facebook. The NASDAQ Composite is often used as a benchmark for the 

The data features on which the clustering is performed includes:
Market Cap: Total market value of a company's outstanding shares, calculated by multiplying the number of outstanding shares by the current market price of one share.

Book/sh: Book value per share, calculated by dividing the company's total equity by the number of outstanding shares.

 P/E: Price-to-earnings ratio, calculated by dividing the current market price per share by the company's earnings per share (EPS).

 P/S: Price-to-sales ratio, calculated by dividing the current market price per share by the company's revenue per share.

 P/B: Price-to-book ratio, calculated by dividing the current market price per share by the company's book value per share.

 ROE: Return on equity, a measure of a company's profitability calculated by dividing its net income by its shareholder equity.

 ROI: Return on investment, a measure of a company's profitability calculated by dividing its net income by its total assets.

 EPS (ttm): Earnings per share for the most recent twelve-month period.

 Debt/Eq: Debt-to-equity ratio, a measure of a company's financial leverage calculated by dividing its total liabilities by its shareholder equity.

 Ideal Price Ratios:The ideal price ratios for a good stock can vary depending on the industry and company's growth stage. Generally, a lower P/E ratio indicates that a stock is undervalued or its earnings are expected to grow in the future. A good P/E ratio can range between 10 to 20, but can be higher for high-growth companies. The ideal P/B ratio depends on the industry, but generally, a ratio less than 1 is considered undervalued. The P/S ratio should also be analyzed relative to industry peers, but a good ratio would be less than 1. A high ROE and ROI is usually good indicator of a well-managed company, but the ideal ratio can vary depending on the industry. Finally, a lower Debt/Equity ratio is usually better, with a ratio less than 1 is considered good.

Once the above data is run over the K-means, for k clusters,
we select the best cluster by taking into account the
ROI and run our algorithms for these companies. 

Stock Data 
From the above approach, once the tickers are finalized,
we will download the historical Data For Stocks from 
yahoo finance. We will use 10 years of history data.
The data contains the following features:
 High - Represents the highest price at which the stock was traded during a particular day.
 Low - represents the lowest price at which the stock was traded during a particular day.
 Open represents the opening price of the stock at the beginning of a particular day.
 Close - represents the closing price of the stock at the end of a particular day.
 Volume - represents the total number of shares of the stock that were traded during a particular day.
 Adj Close - represents the closing price of the stock adjusted for any corporate actions, such as stock splits or dividends, that occurred during the day.
Once the data is ready, we will run the algorithms to predict the target class i.e whether the market will increase or decrease based on the historical data.

 Stock News Data 

The second part of the project involves conducting sentiment analysis on news headlines to assess their impact on stock market trends. After selecting the tickers/companies, we will use the Beautiful Soup library to scrape news headlines from Finviz over the past three days. We will store the headlines for each company in a data frame, which includes a date column and a string containing the news headlines data.

To obtain historical data, one needs to buy a premium account, while the concept of stale news also comes into play. Since the stock market is highly volatile, we do not require excessive amounts of data to predict its behavior the following day. However, we will also use a dataset from Kaggle to train our algorithm. This dataset includes features such as :
 Date: the date on which the headlines were published
 Headlines: Top 25 news stories for a particular company.
 Label: A label containing values 1/0/2  indicating whether the market increased, decreased, or remained stable i.e neutral.


We will use the Kaggle dataset for training purposes and live data from Finviz for testing purposes. Once the data is available, we will apply various natural language processing techniques and perform sentiment analysis. This analysis will help us predict whether the news is positive or negative and assign a corresponding class. Based on the previous day's news headlines, we will track market behavior accordingly.


 Stock Twitter Data

The third part of the project involves conducting sentiment analysis on Twitter data to assess their impact on stock market trends. After selecting the tickers/companies, we will use the Snscrape library to scrape news headlines from Twitter. We will store the tweet for each company in a data frame, which includes a date column and a string containing the Tweet and the username.
The dataset contains the following features:

Once the data is available, we apply various natural language processing techniques and perform sentiment analysis.


Algorithms
Predicting Stock Market

A target variable is created that the machine learning model will predict. This target variable is likely to be the future stock price of each of the three companies.
The pipeline is a series of different classification models. Each model has been optimized using Grid Search Cv, which is a method to find the best hyperparameters (settings) for each model.
A Voting Classifier is used to combine the predictions of multiple machine learning models. It works by taking the majority vote of each model's prediction. This can often lead to better overall accuracy than using a single model.
The precision score is used to evaluate the performance of a classification model. It measures the percentage of correct positive predictions out of all positive predictions. A higher precision score indicates better performance.
Finally, the precision scores for each company (Amazon, Apple, and Google) are reported. These scores indicate how well the machine learning model is able to predict the stock price for each company. A precision score of 0.68 for Amazon, 0.63 for Apple, and 0.65 for Google suggests that the model is fairly accurate in predicting the stock prices of these companies.


Algorithm for News Sentiment Analysis

 The first step is to apply sentiment analysis on the news articles.The sentiment analyzer used in this case is a pre-trained Bidirectional Encoders from transformers model. It takes use of Transformer, an attention mechanism that discovers contextual relationships between words (or sub-words) in a text. Transformer comes with two independent processes in its basic configuration: an encoder that reads the text input and a decoder that creates a task prediction. The Transformer encoder reads the entire string of words at once, in contrast to directional models, which read the text input sequentially (from right to left or left to right).

 The output of the sentiment analyzer is given in the form of stars. A single star with a compound score closer to 1 signifies a strong negative sentiment, while five stars with a compound score closer to 1 signifies a strong positive sentiment. The compound score is a metric used to represent the overall sentiment of the text, with values closer to 1 indicating a stronger sentiment.

The next step is to aggregate the sentiment scores for all the news articles related to the particular stock on a given day. This means that the sentiment scores of each individual news article are combined to give an overall sentiment score for the stock on that day. This aggregation can be done by taking the average of the sentiment scores or by using a weighted average based on the importance of each news article.

Algorithm for Tweet Sentiment Analysis
Snscrape is a Python package used for scraping social media platforms such as Twitter, Tumblr, Reddit, and others. It allows to scrape tweets and other social media content based on certain criteria, such as hashtags, keywords, and specific user accounts.

Flair is a Python package used for NLP tasks such as text classification, named entity recognition, and sentiment analysis. Flair's sentiment analysis model is based on a convolutional neural network (CNN) and a bidirectional long short-term memory (BiLSTM) architecture, which makes it highly accurate and efficient.

Once the tweets have been scraped and saved using Snscapre, we then use Flair to perform sentiment analysis on the text data. We pass the scraped tweets through the model to classify them as either positive, negative, or neutral.

Experiments and Results

 Once the closing price has been obtained, it can be fed into the WebApp that uses a model to predict whether the stock is likely to experience an upward or downward trend in the near future.
 If the WebApp shows a buy signal, then a risk-averse trader may choose to place an order for the stock as soon as the market opens. This means that the trader is buying the stock at the current price, with the expectation that it will increase in value over time.
Based on the model's predictions, the trader can choose to hold on to the stock until the end of the day or longer. If the model is accurate and the stock does experience an upward trend, the trader can sell the stock at a higher price than they bought it for, resulting in a profit.
Before implementing this strategy in a live trading scenario, it is important to test it thoroughly using a stock simulator. This allows the trader to see how the strategy performs under different market conditions and make any necessary adjustments before investing real money.
On 13th April our model predicted a buy signal for Google Stock, i.e. both the News Sentiment Analyzer as well as the Classifier Model, and these are the results of our trade:
We bought the stock and sold the next day to get a 2 percent profit on our capital.
Likewise for the entire week our classifier predicted to short Amazon stock and it worked accurately.

Conclusions
The prediction of stock market movements using machine learning can help investors make informed decisions about buying and selling stocks. The motivation for solving this problem is to provide investors with a competitive edge and achieve better returns on their investments. To solve this problem, historical price data, news headlines and sentiment data are needed. We have collected data from various API's for multiple purposes. For the fundamental indicators we have scraped the Finviz Website. For the historical stock data we have used YahooFinance API. For scraping the news article we have used Finviz Website containing news. For Tweets data we have used an open source library called as Sns Scrape to get tweets.

Once the data has been gathered, the next step is to group similar stocks together using clustering techniques. In this case, stocks with similar market characteristics and performance metrics will be grouped together to create a portfolio of stocks that are expected to perform similarly.

After clustering stocks, the next step is to predict the overall market performance using various predictive models. These models use historical market data and other market indicators to forecast the market direction and identify potential investment opportunities.

In addition to quantitative data analysis, it is also important to analyze market sentiments and opinions through qualitative analysis techniques such as sentiment analysis. This involves analyzing social media posts, news articles, and other sources to understand the overall public sentiment about the market and individual stocks. This information can be used to adjust investment strategies and make informed decisions.

Finally, based on the insights gathered from predicting the market and analyzing market sentiments based on News and Twitter, an investment decision can be made. This decision will involve choosing a portfolio of stocks that are expected to perform well based on market trends, historical data, and sentiment analysis. The investment decision should be based on a well-informed and data-driven approach to maximize returns and minimize risks.



