
# python imports
import requests
import pandas as pd
import io
import sys
import sqlalchemy
import os

# application imports
sys.path.insert(0, os.getcwd())
from common.db_config import DATA_BASE



class Tickers:

    def __init__(self):
        # Connect to the database
        self.data_base_engine = sqlalchemy.create_engine(
f'mysql+mysqlconnector://{DATA_BASE["USER"]}:{DATA_BASE["PASSWORD"]}@{DATA_BASE["HOST"]}:{DATA_BASE["PORT"]}/{DATA_BASE["DATABASE"]}')

    def __del__(self):
        del self.data_base_engine


    def insert_tickers_from_nse(self):
        try:
            stocks_from_database = pd.read_sql(sql="nse_stocks",con=self.data_base_engine)

            nse_url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

            # create Session from 'real' browser
            headers = {
                'User-Agent': 'Mozilla/5.0'
            }
            session = requests.Session()
            session.headers.update(headers)
            # do a get call now
            url = 'https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv'
            response = session.get(nse_url)
            session.close()

            # saving it to pd df for further preprocessing
            df_nse = pd.read_csv(io.BytesIO(response.content))
            ticker_list_from_database = stocks_from_database["SYMBOL"].unique().tolist()
            ticker_list_from_nse = df_nse["SYMBOL"].unique().tolist()
            newly_listed_tickers = list(set(ticker_list_from_nse).difference(set(ticker_list_from_database)))
            if len(newly_listed_tickers) != 0:
                df_nse.to_sql(name='nse_stocks', con=self.data_base_engine, if_exists='append', index=False)
            # BSE data download
            # download_dir = "E:/Shrimandhar/stock_automation/pythonProject/temp_files"
            # bse_link = 'https://mock.bseindia.com/corporates/List_Scrips.html'
            #
            # # change download directory if required
            # prefs = {"download.default_directory": download_dir}
            #
            # options = webdriver.ChromeOptions()
            # options.add_experimental_option("prefs", prefs)
            #
            # # intiate browser
            # browser = Browser('chrome', options=options, headless=True)
            #
            # # visit link
            # browser.visit(bse_link)
            #
            # # fill out form fields
            # browser.find_by_id('ddlsegment').select("Equity")
            # browser.find_by_id('ddlstatus').select("Active")
            #
            # # hit submit button
            # browser.find_by_id('btnSubmit').click()
            #
            # # let the table load
            # browser.is_element_present_by_text("Issuer Name")
            # time.sleep(5)
            #
            # # download
            # browser.find_by_id('lnkDownload').click()
            #
            # df_bse = pd.read_csv(os.path.join(download_dir, "Equity.csv"))
            #
            # # some columns naming convention was different
            # df_bse = df_bse.rename(columns={
            #     "Security Name": "NAME OF COMPANY",
            #     "Security Id": "SYMBOL"
            # })
            #
            # # merged on SYMBOL
            # final_df = pd.merge(df_nse, df_bse, on='SYMBOL', how="outer")
            return
        except Exception as e:
            print(e, sys.exc_info()[2])


if __name__ == "__main__":
    tickers = Tickers()
    tickers.insert_tickers_from_nse()