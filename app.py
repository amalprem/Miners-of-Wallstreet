from flask import Flask, render_template,request
from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf #
import pandas_ta as ta #
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import snscrape.modules.twitter as sntwitter
from datetime import date, timedelta
import re
import flair

def best_sharpe_ratio():
    import pandas as pd
    from yahoo_fin import stock_info as si
    import pandas as pd
    import numpy as np
    from bs4 import BeautifulSoup as soup
    from urllib.request import Request, urlopen
    import time
    def get_tickers():
        sp500_tickers = set(si.tickers_sp500())
        dow_tickers = set(si.tickers_dow())
        nasdaq_tickers = set()
        print(sp500_tickers)
        print(dow_tickers)

        # join the 3 sets into one. Because it's a set, there will be no duplicate symbols
        symbols = set.union(sp500_tickers, nasdaq_tickers, dow_tickers)
        my_list = ['W', 'R', 'P', 'Q']
        del_set = set()
        sav_set = set()

        for symbol in symbols:
            if len(symbol) > 4 and symbol[-1] in my_list:
                del_set.add(symbol)
            else:
                sav_set.add(symbol)
        return list(sav_set)


    #tickers=['ISRG', 'NDAQ', 'KLAC', 'WYNN', 'ETSY', 'ALL', 'KMX', 'FOX', 'TSCO', 'NWS', 'EXPE', 'DPZ', 'EW', 'EVRG', 'TRMB', 'GOOG', 'VRTX', 'CCL', 'TSLA', 'GEHC', 'CMS', 'AEP', 'TSN', 'STX', 'SYF', 'WFC', 'SCHW', 'PKG', 'BWA', 'CME', 'DVA', 'LYV', 'GIS', 'OTIS', 'TTWO', 'GNRC', 'EMR', 'CMA', 'AVB', 'CHD', 'EQT', 'NOC', 'MNST', 'ARE', 'CZR', 'FRC', 'ADI', 'FE', 'HLT', 'RJF', 'SEE', 'JPM', 'AMZN', 'GOOGL', 'IR', 'SYK', 'BLK', 'WELL', 'EL', 'ROST', 'QRVO', 'TXN', 'ALGN', 'TRV', 'BRK-B', 'DD', 'KEY', 'HES', 'UNH', 'DHI', 'MDLZ', 'VMC', 'MAR', 'DISH', 'KMB', 'CRL', 'BSX', 'IDXX', 'CPT', 'FFIV', 'MDT', 'NCLH', 'ACN', 'IFF', 'AON', 'DTE', 'INCY', 'TAP', 'BG', 'MGM', 'LYB', 'INTU', 'AXP', 'PAYX', 'NSC', 'PODD', 'TJX', 'ODFL', 'MKC', 'HPE', 'LIN', 'GRMN', 'WEC', 'HRL', 'PG', 'SEDG', 'GM', 'CE', 'QCOM', 'MTD', 'WAT', 'EPAM', 'EFX', 'POOL', 'BBY', 'TFX', 'DUK', 'GE', 'MOS', 'MSI', 'MU', 'DAL', 'LLY', 'BA', 'OXY', 'ANET', 'HOLX', 'MOH', 'MRK', 'XYL', 'T', 'OKE', 'CFG', 'ABC', 'XEL', 'CSCO', 'GD', 'MCD', 'J', 'RHI', 'LNC', 'PXD', 'SWK', 'CNP', 'AIG', 'TFC', 'NKE', 'MAS', 'IBM', 'NOW', 'PAYC', 'SHW', 'MMC', 'CB', 'CRM', 'EXR', 'CAG', 'AAL', 'RF', 'CCI', 'PARA', 'MS', 'TT', 'INTC', 'SWKS', 'HAL', 'BF-B', 'CEG', 'TMUS', 'AKAM', 'KDP', 'PCG', 'MCO', 'BEN', 'XOM', 'WM', 'MKTX', 'EXPD', 'RE', 'ZION', 'WBD', 'BAX', 'NTAP', 'JNPR', 'COST', 'STE', 'ED', 'TYL', 'ALLE', 'DLR', 'DE', 'CDNS', 'PNR', 'PSX', 'ROK', 'PGR', 'HII', 'LVS', 'PEAK', 'UHS', 'AMP', 'LEN', 'CMI', 'IEX', 'ETN', 'PH', 'FISV', 'PM', 'TRGP', 'EA', 'SLB', 'ATVI', 'ETR', 'GWW', 'NVR', 'NVDA', 'CMCSA', 'MCK', 'KHC', 'LDOS', 'WMB', 'HCA', 'ROL', 'HIG', 'NXPI', 'COP', 'EQIX', 'BMY', 'SNPS', 'ANSS', 'SRE', 'IPG', 'IVZ', 'ACGL', 'SJM', 'WY', 'PEG', 'TDG', 'PFG', 'C', 'VTR', 'VFC', 'VRSN', 'DG', 'CBRE', 'LH', 'UPS', 'NWL', 'IP', 'AIZ', 'BK', 'CSGP', 'NFLX', 'NRG', 'PPG', 'DOW', 'MPWR', 'CTSH', 'WRB', 'KEYS', 'APA', 'BKNG', 'CINF', 'CVS', 'CPRT', 'PRU', 'UAL', 'FSLR', 'REG', 'EBAY', 'LW', 'DGX', 'EOG', 'FICO', 'CF', 'BDX', 'D', 'MTB', 'KMI', 'CTLT', 'HST', 'UNP', 'AMT', 'RCL', 'DIS', 'L', 'LKQ', 'LOW', 'KIM', 'PYPL', 'AVGO', 'VZ', 'EQR', 'CNC', 'CDW', 'RTX', 'AMGN', 'ZTS', 'ORLY', 'HWM', 'ULTA', 'FTNT', 'HSIC', 'AVY', 'CHTR', 'AAP', 'TECH', 'FAST', 'A', 'FDX', 'NI', 'HON', 'MMM', 'CHRW', 'MHK', 'CI', 'GL', 'WDC', 'ADSK', 'DHR', 'ABBV', 'ICE', 'COF', 'PEP', 'USB', 'WMT', 'CMG', 'PNW', 'ENPH', 'BAC', 'MSCI', 'RSG', 'WAB', 'SPG', 'FMC', 'KR', 'APH', 'HD', 'GS', 'HUM', 'PFE', 'HSY', 'ALK', 'TER', 'ESS', 'DOV', 'VICI', 'MO', 'MRO', 'ZBH', 'SPGI', 'ATO', 'BRO', 'ECL', 'HBAN', 'GLW', 'ADP', 'AJG', 'FIS', 'AMCR', 'FLT', 'REGN', 'CSX', 'ZBRA', 'VRSK', 'PCAR', 'PPL', 'APD', 'TMO', 'O', 'CLX', 'F', 'COO', 'DRI', 'META', 'PTC', 'MAA', 'BBWI', 'AOS', 'EXC', 'V', 'WBA', 'CVX', 'MTCH', 'HPQ', 'JNJ', 'KO', 'MET', 'AWK', 'MLM', 'FANG', 'CAT', 'NWSA', 'STZ', 'UDR', 'LHX', 'XRAY', 'URI', 'GILD', 'INVH', 'RL', 'SNA', 'TPR', 'PLD', 'PHM', 'EMN', 'NUE', 'BKR', 'OGN', 'HAS', 'TXT', 'GEN', 'FCX', 'FOXA', 'AMD', 'LUV', 'WRK', 'NDSN', 'STLD', 'CBOE', 'PWR', 'CARR', 'ILMN', 'BXP', 'MCHP', 'JBHT', 'PKI', 'AMAT', 'NTRS', 'DXCM', 'SYY', 'VLO', 'OMC', 'TDY', 'CTRA', 'DLTR', 'TGT', 'DVN', 'IRM', 'K', 'LMT', 'WTW', 'WST', 'WHR', 'ALB', 'CL', 'MRNA', 'DXC', 'EIX', 'MA', 'DFS', 'VTRS', 'BALL', 'TROW', 'ADBE', 'CTAS', 'AZO', 'FRT', 'FDS', 'YUM', 'MPC', 'CPB', 'BIO', 'BIIB', 'IT', 'NEM', 'CAH', 'SBUX', 'APTV', 'CTVA', 'JCI', 'BR', 'ELV', 'ES', 'ROP', 'TEL', 'FITB', 'PSA', 'SBAC', 'AME', 'FTV', 'MSFT', 'LRCX', 'GPN', 'AES', 'SO', 'AEE', 'NEE', 'AFL', 'ITW', 'PNC', 'LNT', 'IQV', 'GPC', 'ON', 'ORCL', 'RMD', 'JKHY', 'STT', 'ADM', 'AAPL', 'CDAY', 'ABT']
    tickers=['ISRG', 'NDAQ', 'KLAC', 'WYNN', 'ETSY', 'ALL', 'KMX', 'FOX', 'TSCO', 'NWS', 'EXPE', 'DPZ', 'EW', 'EVRG', 'TRMB', 'GOOG', 'VRTX', 'CCL', 'TSLA', 'GEHC', 'CMS', 'AEP', 'TSN', 'STX', 'SYF','AMZN']
    dfs = []

    for symbol in tickers:
        try:
            url = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
            print(url)
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
            html = soup(webpage, "html.parser")
            fundamentals = pd.read_html(str(html), attrs={'class': 'snapshot-table2'})[0]
            fundamentals.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
            attribute = []
            l1 = len(fundamentals)
            for k in np.arange(0, l1, 2):
                attribute.append(fundamentals[f'{k}'])
            df_column_attribute = pd.concat(attribute, ignore_index=True)
            values = []
            l2 = len(fundamentals)
            for k in np.arange(1, l2, 2):
                values.append(fundamentals[f'{k}'])
            df_column_values = pd.concat(values, ignore_index=True)
            fundamental_dataframe = pd.DataFrame()
            fundamental_dataframe['Attributes'] = df_column_attribute
            fundamental_dataframe['Values'] = df_column_values
            fundamental_dataframe = fundamental_dataframe.set_index('Attributes')

            # Append the dataframe for the current ticker to the list of dataframes
            dfs.append(fundamental_dataframe)
            time.sleep(0.1)

        except Exception as e:
            print(f"An error occurred while processing {symbol}: {e}")
    df = pd.concat(dfs, axis=1, keys=tickers)
    df = df.transpose()
    df_final = df[['Market Cap', 'Book/sh', 'P/E', 'P/S', 'P/B', 'ROE', 'ROI', 'EPS (ttm)', 'Debt/Eq']]
    df_final1 = df_final.reset_index()
    df_final1.drop(['level_1', 'Market Cap'], axis=1, inplace=True)
    df_fundamental = df_final1
    df_fundamental = df_fundamental.rename(columns={'level_0': 'Tickers'})
    df_fundamental['Book/sh'] = pd.to_numeric(df_fundamental['Book/sh'], errors='coerce')
    df_fundamental['P/E'] = pd.to_numeric(df_fundamental['P/E'], errors='coerce')
    df_fundamental['P/S'] = pd.to_numeric(df_fundamental['P/S'], errors='coerce')
    df_fundamental['P/B'] = pd.to_numeric(df_fundamental['P/B'], errors='coerce')
    df_fundamental['ROE'] = pd.to_numeric(df_fundamental['ROE'].str.rstrip('%'), errors='coerce') / 100
    df_fundamental['ROI'] = pd.to_numeric(df_fundamental['ROI'].str.rstrip('%'), errors='coerce') / 100
    df_fundamental['EPS (ttm)'] = pd.to_numeric(df_fundamental['EPS (ttm)'], errors='coerce')
    df_fundamental['Debt/Eq'] = pd.to_numeric(df_fundamental['Debt/Eq'], errors='coerce')
    df_class = df_fundamental['Tickers']
    df_k_means_cluster = pd.DataFrame()
    list_col = ['Book/sh', 'P/E', 'P/S', 'ROE', 'ROI', 'EPS (ttm)', 'Debt/Eq']
    for column in list_col:
        df_k_means_cluster[column] = df_fundamental[column]
    print("hello my code is here")
    print(df_fundamental.shape)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_k_means_cluster_scaled = scaler.fit_transform(df_k_means_cluster)
    df_k_means_cluster_scaled = pd.DataFrame(df_k_means_cluster_scaled, columns=df_k_means_cluster.columns)
    df_k_means_cluster_scaled.fillna(0, inplace=True)
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore', message='The default value of `n_init`')
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df_k_means_cluster_scaled)
        wcss.append(kmeans.inertia_)
    # plt.plot(range(1, 11), wcss)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('WCSS')


    # Perform K-means clustering with the optimal number of clusters
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(df_k_means_cluster_scaled)
    df_k_means_cluster['Cluster'] = kmeans.labels_
    pca = PCA(n_components=2)
    pca.fit(df_k_means_cluster_scaled)
    df_pca = pca.transform(df_k_means_cluster_scaled)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = kmeans.labels_
    # plt.figure(figsize=(8, 6))
    # plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis')
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.title('K-means Clustering Results')

    df_k_means_cluster['Ticker'] = df_fundamental['Tickers']
    cluster_stats = df_k_means_cluster.groupby('Cluster')['ROI'].agg(['mean', 'std'])
    risk_free_rate = 0.02
    cluster_stats['sharpe_ratio'] = (cluster_stats['mean'] - risk_free_rate) / cluster_stats['std']
    grouped_df = df_k_means_cluster.groupby('Cluster').agg({'Ticker': lambda x: list(x)}).reset_index()
    best_cluster = cluster_stats['sharpe_ratio'].idxmax()
    best_cluster_list = grouped_df[grouped_df['Cluster'] == best_cluster]['Ticker'].to_list()
    # best_cluster_list=[]
    # best_cluster_list.append('AMZN')
    # best_cluster_list.append('MFST')
    # best_cluster_list.append('TSLA')
    return best_cluster_list

def predict_stock_rf(stock_ticker):
    import pandas as pd
    import numpy as np
    import datetime as dt
    import yfinance as yf
    import pandas_ta as ta
    print("Imported the libraries in the second function")
    def sma(df,window,feature):
        return df[feature].rolling(window=window).mean()

    def wilder(df,window,feature):
        return ta.wma(df[feature], window)

    def atr(df,window,high,low,close):
        return ta.atr(df[high], df[low], df[close], length=window)

    def adx(df,window,high,low,close):
        return ta.adx(df[high], df[low], df[close], length=window)


    def stochastic_oscillator(df,w,high, low, close, k):
        a = ta.momentum.stoch(df[high], df[low], df[close], window=k, smooth_window=w)
        a.columns=['K','D']

        return sma(a,w,'K')

    def rsi(df,w,close):
        a = ta.momentum.rsi(df[close]).fillna(0)
        df=pd.DataFrame(a.values)
        df.columns=['rsi']
        a=sma(df,w,'rsi')

        return a

    def bollinger_bands(df,w,close):
        a=pd.DataFrame(df['Close'],columns=['Close'])
        b=a.ta.bbands(length=w, std=2, append=True)
        b.columns=['BBL','BBM','BBU','BBB','BBP']
        return b[['BBL','BBU']]

    stock_ticker=str(stock_ticker)
    features=['ticker','Open','High','Low','Close','Volume']
    data=pd.DataFrame()
    print("the ticker i ma working with is {}".format(stock_ticker))



    hist = yf.Ticker(stock_ticker).history(period='30y')
    hist['ticker']=stock_ticker
    # Generating SMA for close and volume
    hist['SMA_2_close']= sma(hist,2,'Close')
    hist['SMA_5_close']= sma(hist,5,'Close')
    hist['SMA_2_volume']= sma(hist,2,'Volume')
    hist['SMA_5_volume']= sma(hist,5,'Volume')
    # Genrating ATR
    hist['ATR_2']=atr(hist,2,'High','Low','Close')
    hist['ATR_5']=atr(hist,5,'High','Low','Close')
    # Generating ADX
    hist[['ADX_2','DMP_2','DMN_2']]=adx(hist,2,'High','Low','Close')
    hist[['ADX_5','DMP_5','DMN_5']]=adx(hist,5,'High','Low','Close')
    # Generating Stochastic oscilators d
    hist['Stochastic_2_D']= stochastic_oscillator(hist,2,'High','Low','Close', 14)
    hist['Stochastic_5_D']= stochastic_oscillator(hist,5,'High','Low','Close', 14)

    # Generating RSI
    hist['RSI_2']=rsi(hist,2,'Close').to_list()
    hist['RSI_5']=rsi(hist,5,'Close').to_list()
    #Generating Bollinger BAnds
    hist[['BBL_2','BBU_2']]=bollinger_bands(hist,2,'Close')
    hist[['BBL_5','BBU_5']]=bollinger_bands(hist,5,'Close')
    hist['Close_shifted']=hist['Close'].transform(lambda x: x.shift(-1))
    hist.dropna(inplace=True)

    hist['Output']=100*(hist['Close_shifted']-hist['Open'])/hist['Open']
    hist['Target']= np.where(hist['Output']>0,1,0)
    # can winsorize
    data=pd.concat([data,hist])

    data.dropna(inplace=True)
    data.drop(columns=['Dividends','Stock Splits'],inplace=True)
    data_sample=data.head(1)
    dictionary_df = data_sample.to_dict(orient='records')
    return dictionary_df

def news_analyser(stock_ticker):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    print("I am in the news function")
    def create_dict(a, b):
        result_dict = {}
        result_dict[a] = b
        return result_dict

    def news_classifier(rating):
        if rating < 2.5:
            return "Today's day was bad according to the news"
        elif rating > 2.5 and rating < 3:
            return "The new's for the stock was neutral and we need to look at other indicators"
        else:
            return "Stock news are on the positive side of spectrum for today"

    def weighted_average(input_dict):
        numerator = sum(key * value for key, value in input_dict.items())
        denominator = sum(input_dict.values())
        weighted_avg = numerator / denominator
        return weighted_avg

    from urllib.request import urlopen, Request
    from bs4 import BeautifulSoup
    import pandas as pd
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    sentiment_classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
    finviz_url = 'https://finviz.com/quote.ashx?t='
    ticker = stock_ticker
    news_tables = {}
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, 'html')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

    parsed_data = []
    for ticker, news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            if row.a and row.a.text:
                title = row.a.get_text()
                date_data = row.td.text.split(' ')
                if len(date_data) == 1:
                    time = date_data[0]
                else:
                    date = date_data[0]
                    time = date_data[1]

                result = sentiment_classifier(title)
                score = result[0]['score']
                label = result[0]['label']

                parsed_data.append([ticker, date, time, title, score, label])

    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title', 'score', 'label'])
    df['label'] = df['label'].apply(lambda x: int(x.split(' ')[0]))
    df['date'] = pd.to_datetime(df['date'])
    grouped_by_date = df.groupby('date').agg({'score': 'mean', 'label': lambda x: x.value_counts().to_dict()})
    grouped_by_date = grouped_by_date.reset_index()
    grouped_by_date['Overall_rating'] = grouped_by_date['label'].apply(weighted_average)
    grouped_by_date['News_group'] = grouped_by_date['Overall_rating'].apply(news_classifier)
    result_dict = grouped_by_date[['date', 'News_group']].apply(
        lambda row: {row['date'].strftime('%Y-%m-%d'): row['News_group']}, axis=1).to_dict()
    result_dict = {row['date'].strftime('%Y-%m-%d'): {'News_group': row['News_group']} for _, row in
                   grouped_by_date.iterrows()}
    formattedData = []
    for date, news_group in result_dict.items():
        dict_item = {
            'date': date,
            'data': news_group['News_group']
        }
        formattedData.append(dict_item)
    return formattedData

def fetch_tickers(index):
    print('fetch_tickers method called')
    allTickers = []
    from yahoo_fin import stock_info as si
    check_list = ['W', 'R', 'P', 'Q']
    del_set = set()
    sav_set = set()
    match index:
        case "DOW_JONES":
            ticker_var = set(si.tickers_dow())
            for stock in ticker_var:
                if len(stock) > 4 and stock[-1] in check_list:
                    del_set.add(stock)
                else:
                    sav_set.add(stock)
            allTickers = list(sav_set)            
            return allTickers
        case "FTS_100":
            ticker_var = set(si.tickers_ftse100())
            for stock in ticker_var:
                if len(stock) > 4 and stock[-1] in check_list:
                    del_set.add(stock)
                else:
                    sav_set.add(stock)
            allTickers = list(sav_set)            
            return allTickers 
        case "NASDAQ":
            ticker_var = set(si.tickers_nasdaq())
            for stock in ticker_var:
                if len(stock) > 4 and stock[-1] in check_list:
                    del_set.add(stock)
                else:
                    sav_set.add(stock)
            allTickers = list(sav_set)            
            return allTickers
        case "NIFTY50":
            print('inside NIFTY50 case')
            ticker_var = set(si.tickers_nifty50(include_company_data = True))
            print(ticker_var)
            for stock in ticker_var:
                if len(stock) > 4 and stock[-1] in check_list:
                    del_set.add(stock)
                else:
                    sav_set.add(stock)
            allTickers = list(sav_set)            
            return allTickers
        case default:
            print('inside default case')
            ticker_var = set(si.tickers_other())
            for stock in ticker_var:
                if len(stock) > 4 and stock[-1] in check_list:
                    del_set.add(stock)
                else:
                    sav_set.add(stock)
            allTickers = list(sav_set)
            return allTickers


# Stock Data Scrape
today = date.today()

def company_tweets(company):

    def scrape_tweets(company):
        # Search query
        query = '"' + company + ' Stock" lang:en until:' + today.strftime("%Y-%m-%d") + ' since:2023-01-01 filter:verified'
        tweets = []
        limit = 10000

        # Retrieval of tweets
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            if len(tweets) == limit:
                break
            else:
                tweets.append([tweet.date, tweet.user.username, tweet.rawContent])
        return tweets

        
    def preprocess_tweet(text):
        processed_tweet = []
        text = text.lower()
        
        #Clean only digits
        text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
        
        # Replaces URLs with the word URL
        text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', text)
        
        # Replace @handle with the word USER_MENTION
        text = re.sub(r'@[\S]+', '', text)
        
        # Replace # with the hashtag
        text = re.sub(r'#(\S+)', '', text)
        
        # Remove RT (retweet)
        text = re.sub(r'\brt\b', '', text)
        
        # Replace 2+ dots with space
        text = re.sub(r'\.{2,}', ' ', text)
        
        # Strip space, " and ' from tweet
        text = text.strip(' "\'')
    
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Custom chars
        text = text.replace('₺','')
        text = text.replace('=','')
        text = text.replace('’','')
        text = text.replace('|','')
        text = text.replace('‘','')
        text = text.replace('/','')
        text = text.replace('…','')
        text = text.replace('–','')
        text = text.replace('&','')
        text = text.replace('“','')
        text = text.replace('”','')
        text = text.replace('+','')
        text = text.replace('%','')
        text = text.replace('@','')
        text = text.replace('#','')

        return text


    def sentiment_analysis(dataset):

        probs = []
        sentiments = []

        sentiment_model = flair.models.TextClassifier.load('en-sentiment')

        for tweet in dataset['Tweet'].to_list():
            tweet = preprocess_tweet(tweet)
            sentence = flair.data.Sentence(tweet)
            sentiment_model.predict(sentence)
            # Extracting sentiments
            probs.append(sentence.labels[0].score)  
            sentiments.append(sentence.labels[0].value)  

        # Adding probabilities and sentiments to dataframe
        dataset['Probability'] = probs
        dataset['Sentiment'] = sentiments
        return dataset

    df = pd.DataFrame(scrape_tweets(company), columns=['Date', 'User', 'Tweet'])
    df = sentiment_analysis(df)

    # Sentiment for today
    ndf = df.copy()
    ndf['Date'] = pd.to_datetime(ndf["Date"]).dt.date
    ndf['Date'] = ndf['Date'].astype('string')
    if ndf.loc[ndf['Date'] == (date.today() - timedelta(days = 1)).strftime("%Y-%m-%d")].size != 0:
        temp = ndf.loc[ndf['Date'] == (date.today() - timedelta(days = 1)).strftime("%Y-%m-%d")]['Sentiment'].mode()
        return ''.join(temp.values)
    else:
        return 'NEUTRAL'


def tweet_sentiment(ticker):
    match ticker:
        case 'AMZN':
            return company_tweets('Amazon')
        case 'AAPL':
            return company_tweets('Apple')
        case 'GOOG':
            return company_tweets('Google')
        case default:
            return 'NEUTRAL'

def model_predictions(ticker):
    def sma(df,window,feature):
        #use interpolate  or ffill to handle nan
        return df[feature].rolling(window=window).mean()

    def wilder(df,window,feature):
        return ta.wma(df[feature], window)

    def atr(df,window,high,low,close):
        return ta.atr(df[high], df[low], df[close], length=window)

    def adx(df,window,high,low,close):
        return ta.adx(df[high], df[low], df[close], length=window)


    def stochastic_oscillator(df,w,high, low, close, k):
        a = ta.momentum.stoch(df[high], df[low], df[close], window=k, smooth_window=w)
        a.columns=['K','D']

        return sma(a,w,'K')

    def rsi(df,w,close):
        a = ta.momentum.rsi(df[close]).fillna(0)
        df=pd.DataFrame(a.values)
        df.columns=['rsi']
        a=sma(df,w,'rsi')

        return a

    def bollinger_bands(df,w,close):
        a=pd.DataFrame(df['Close'],columns=['Close'])
        b=a.ta.bbands(length=w, std=2, append=True)
        b.columns=['BBL','BBM','BBU','BBB','BBP']
        return b[['BBL','BBU']]



    hist = yf.Ticker(ticker).history(period="1Y")
    hist['ticker']= ticker
    # Generating SMA for close and volume
    hist['SMA_2_close']= sma(hist,2,'Close') 
    hist['SMA_5_close']= sma(hist,5,'Close') 
    hist['SMA_2_volume']= sma(hist,2,'Volume') 
    hist['SMA_5_volume']= sma(hist,5,'Volume') 
    # Genrating ATR
    hist['ATR_2']=atr(hist,2,'High','Low','Close')
    hist['ATR_5']=atr(hist,5,'High','Low','Close')
    # Generating ADX
    hist[['ADX_2','DMP_2','DMN_2']]=adx(hist,2,'High','Low','Close')
    hist[['ADX_5','DMP_5','DMN_5']]=adx(hist,5,'High','Low','Close')
    # Generating Stochastic oscilators d
    hist['Stochastic_2_D']= stochastic_oscillator(hist,2,'High','Low','Close', 14)
    hist['Stochastic_5_D']= stochastic_oscillator(hist,5,'High','Low','Close', 14)

    # Generating RSI
    hist['RSI_2']=rsi(hist,2,'Close').to_list()
    hist['RSI_5']=rsi(hist,5,'Close').to_list()
    #Generating Bollinger BAnds
    hist[['BBL_2','BBU_2']]=bollinger_bands(hist,2,'Close')
    hist[['BBL_5','BBU_5']]=bollinger_bands(hist,5,'Close')
    hist['Close_shifted']=hist['Close'].transform(lambda x: x.shift(-1))
    hist.fillna(0)

    hist['Output']=100*(hist['Close_shifted']-hist['Open'])/hist['Open']
    hist['Target']= np.where(hist['Output']>0,1,0)

    hist.fillna(0)
    hist.drop(columns=['Dividends','Stock Splits'],inplace=True)
    test_data= hist.tail(1)
    features=['Open', 'High', 'Low', 'Close', 'Volume','SMA_2_close',
       'SMA_5_close', 'SMA_2_volume', 'SMA_5_volume', 'ATR_2', 'ATR_5',
       'ADX_2', 'DMP_2', 'DMN_2', 'ADX_5', 'DMP_5', 'DMN_5', 'Stochastic_2_D',
       'Stochastic_5_D', 'RSI_2', 'RSI_5', 'BBL_2', 'BBU_2', 'BBL_5', 'BBU_5']
    
    match ticker:
        case 'AMZN':
            with open('./models/trained_models_amzn.pkl', 'rb') as f:
                models = pickle.load(f)
            model_to_be_ran={}
            for key,value in models.items():
                if key=='BaggingClassifier':
                    continue
                else:
                    model_to_be_ran[key]=value
            
            # Assume you have new data in a pandas DataFrame named 'new_data'
            # Make predictions using all the models
            output_values = [model.predict(test_data[features]) for model in model_to_be_ran.values() ]
            combined_output = np.array(output_values).T
            output_df = pd.DataFrame(combined_output, columns=model_to_be_ran.keys())
            
            # Use the bagging classifier to get the final prediction
            bagging = models['BaggingClassifier']
            bagging_output = bagging.predict(output_df)
            return bagging_output[0]
        case 'AAPL':
            with open('./models/trained_models_aapl.pkl', 'rb') as f:
                models = pickle.load(f)
            model_to_be_ran={}
            for key,value in models.items():
                if key=='BaggingClassifier':
                    continue
                else:
                    model_to_be_ran[key]=value
            
            # Assume you have new data in a pandas DataFrame named 'new_data'
            # Make predictions using all the models
            output_values = [model.predict(test_data[features]) for model in model_to_be_ran.values() ]
            combined_output = np.array(output_values).T
            output_df = pd.DataFrame(combined_output, columns=model_to_be_ran.keys())
            
            # Use the bagging classifier to get the final prediction
            bagging = models['BaggingClassifier']
            bagging_output = bagging.predict(output_df)
            return bagging_output[0]
        case 'GOOG':
            with open('./models/trained_models_goog.pkl', 'rb') as f:
                models = pickle.load(f)
            model_to_be_ran={}
            for key,value in models.items():
                if key=='BaggingClassifier':
                    continue
                else:
                    model_to_be_ran[key]=value
            
            # Assume you have new data in a pandas DataFrame named 'new_data'
            # Make predictions using all the models
            output_values = [model.predict(test_data[features]) for model in model_to_be_ran.values() ]
            combined_output = np.array(output_values).T
            output_df = pd.DataFrame(combined_output, columns=model_to_be_ran.keys())
            
            # Use the bagging classifier to get the final prediction
            bagging = models['BaggingClassifier']
            bagging_output = bagging.predict(output_df)
            return bagging_output[0]

def investGuru(response):
    score = 'gray'
    newsScore = 1 if ("positive" in response['news_output']['data']) else 0
    tweetScore = 1 if ("POSITIVE" in response['tweet_sentiment']) else 0
    modelScore = 1 if ("Yes" in response['model_pred']) else 0
    total = newsScore + tweetScore + modelScore
    score = 'green' if total >= 2 else 'red' 
    return score

from flask import Flask, render_template, jsonify

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
@cross_origin()
def index():
    # Render the HTML template
    return "Test Call success"
    # return render_template('./stocks.html')

@app.route('/best-stocks')
def best_stocks():
    # Call the best_sharpe_ratio function to get the list of stocks
    stocks = best_sharpe_ratio()

    # Return the list of stocks as a JSON response
    return jsonify(stocks)

@app.route('/select-tickers/<string:index>')
@cross_origin()
def enlistTickers(index):
    print('Inside fetch tickers')
    # data = request.args
    # print(data)
    # param_args = '' if data[index] == "" else index
    tickers = fetch_tickers(index)
    response = {
        'status': 'success',
        'results': tickers
    }
    return response

@app.route('/predict-stock/<string:stock>')
@cross_origin()
def predict_stock(stock):
    news_output = news_analyser(stock)
    flag = model_predictions(stock)
    tweet_status = tweet_sentiment(stock)
    response = {
        'stock': stock,
        'news_output': news_output[-1],
        'model_pred': 'Yes' if flag else 'No',
        'tweet_sentiment': tweet_status
    }
    response['final_score']=investGuru(response)
    return response

if __name__ == '__main__':
    app.run()
