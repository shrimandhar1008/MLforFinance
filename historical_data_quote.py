from datetime import datetime

#python imports
import yfinance as yf
import pandas as pd
import sqlalchemy
import sys
import os
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import requests_cache
import datetime
from pandas.tseries.offsets import MonthEnd
import time

# application imports
sys.path.insert(0, os.getcwd())
from common.db_config import DATA_BASE
# end date data is excluded

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("yfinance.cache"),
)

class Prices:

    def __init__(self):
        # Connect to the database
        self.data_base_engine = sqlalchemy.create_engine(
            f'mysql+mysqlconnector://{DATA_BASE["USER"]}:{DATA_BASE["PASSWORD"]}@{DATA_BASE["HOST"]}:{DATA_BASE["PORT"]}/{DATA_BASE["DATABASE"]}')

    def __del__(self):
        del self.data_base_engine


    def get_current_date(self):
        return datetime.datetime.today()

    def insert_historical_prices(self):
        try:
            session = requests_cache.CachedSession('yfinance.cache')
            session.headers['User-agent'] = 'my-program/1.0'
            stock_info_from_database_df = pd.read_sql(sql="nse_stocks", con=self.data_base_engine)
            stock_info_from_database_df.columns = [col.strip() for col in stock_info_from_database_df.columns]
            stock_info_from_database_df['DATE OF LISTING'] = pd.to_datetime(stock_info_from_database_df['DATE OF LISTING'])
            stock_info_from_database_df.sort_values(by='DATE OF LISTING', inplace=True)
            date_chunks = pd.date_range('2001-01-31', '2024-12-31',freq='M')[::35]
            stock_list_from_database = stock_info_from_database_df["SYMBOL"].unique().tolist()
            daily_return_stock_list = pd.read_sql("SELECT DISTINCT(ticker) from daily_price_data;", con=self.data_base_engine)['ticker'].tolist()
            stocks_not_present_in_db = list(set(stock_list_from_database).difference(set(daily_return_stock_list)))
            stocks_not_present_in_db = [stock + '.NS' for stock in stocks_not_present_in_db]
            print(datetime.datetime.now())
            for stock in stocks_not_present_in_db:
                start_time = time.time()
                stock_df = pd.DataFrame()
                for idx in range(len(date_chunks)-1):
                    if idx==0:
                        start_date = date_chunks[idx] - MonthEnd(1) + datetime.timedelta(days=1)
                    else:
                        start_date = date_chunks[idx] + datetime.timedelta(days=1)
                    end_date = date_chunks[idx+1] + datetime.timedelta(days=1)
                    stock_data = yf.download(stock, start=start_date.strftime('%Y-%m-%d'),
                                             end=end_date.strftime('%Y-%m-%d'),
                                             back_adjust=False, session=session)
                    if not stock_data.empty:
                        stock_data["ticker"] = stock.split('.')[0]
                        stock_data = stock_data.reset_index()
                        stock_data["Date"] = stock_data["Date"].dt.strftime('%Y-%m-%d')

                        stock_df = pd.concat([stock_df, stock_data], axis=0)
                stock_df = stock_df.reset_index(drop=True)
                stock_df.rename(columns={'Date': 'date','Adj Close': 'Adj_Close'}, inplace=True)
                stock_df.to_sql(name='daily_price_data', con=self.data_base_engine, if_exists='append',
                                  chunksize=1000,
                                  index=False, method='multi')
                print("--- %s seconds ---" % (time.time() - start_time))
            return
        except Exception as e:
            print(e, sys.exc_info()[2])


if __name__ == "__main__":
    prices = Prices()
    prices.insert_historical_prices()