# purpose of this script is to download data FOR THIS TOOL ONLY
import pandas as pd
import yfinance as yf
from pymongo import MongoClient
import logging

class MongoDBConnection:
    def __init__(self, port=27017):
        self.client = MongoClient('localhost', port)
        self.db = self.client['EquityPairTrading']
        self.closecoll = self.db['CloseData']
        self.ohlccoll = self.db['OHLCData']

class DataDownloader:
    def __init__(self, path='GoodTickers.csv'): # otherwise get too many nans
        self.Mongo = MongoDBConnection()
        self.ticker_df = pd.read_csv(path)

    def get_ticker_list(self):
        tickers = self.ticker_df.Tickers.drop_duplicates().to_list()
        return tickers

    def download_and_insert_ohlc_data(self, tickers, start_date=None):
        df1 = yf.download(tickers, threads=True, start=start_date)
        self.raw_download = df1
        df1 = df1.stack().reset_index().dropna()
        df1.rename(columns={'level_1':'Ticker'}, inplace=True)
        self.Mongo.ohlccoll.insert_many(df1.to_dict('records'))

    def insert_close_data(self):
        df1 = self.raw_download.dropna()
        cols = list(df1.columns)
        df2 = df1[[x for x in cols if x[0]=='Close']]
        cols2 = list(df2.columns)
        new_cols = [x[1] for x in cols2]
        df2.columns = new_cols
        dict1 = df2.reset_index().to_dict('records')
        self.Mongo.closecoll.insert_many(dict1)

    def create_indices(self):
        self.Mongo.closecoll.create_index([('Date'),('Ticker')], name='date_ticker')
        self.Mongo.closecoll.create_index([('Date')], name='date')
        self.Mongo.closecoll.create_index([('Ticker')], name='ticker')
        self.Mongo.ohlccoll.create_index([('Date')], name='date')

    def delete_values(self, start_date=None):
        filter_condition = {}
        if start_date:
            filter_condition['Date'] = {'$gte': start_date}
        logging.info(str(filter_condition))
        self.Mongo.closecoll.coll.delete_many(filter_condition)
        self.Mongo.ohlccoll.coll.delete_many(filter_condition)


if __name__ == '__main__':
    dd = DataDownloader()
    tickers = dd.get_ticker_list()
    dd.delete_values()
    dd.download_and_insert_ohlc_data(tickers, start_date="2011-10-13") # otherwise get too many nans
    dd.insert_close_data()
    dd.create_indices()