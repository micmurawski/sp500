import requests
import scrapy
import urllib.parse

import scrapy
import pandas as pd
from collections.abc import Iterable
from scrapy.http import Request, Response
import re
import json
from scrapy.utils.response import get_meta_refresh
import os


QUATERLY_REPORTS = False

CUSTOM_SETTINGS = dict(
    TWISTED_REACTOR="twisted.internet.asyncioreactor.AsyncioSelectorReactor",
    USER_AGENT='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
    RETRY_TIMES=5,
    DOWNLOAD_DELAY=1,
    RANDOMIZE_DOWNLOAD_DELAY=True,
    COOKIES_ENABLED=True,
    CONCURRENT_REQUESTS=4,
    CONCURRENT_REQUESTS_PER_DOMAIN=4,
    ROBOTSTXT_OBEY=False,
    # LOG_STDOUT = True,
    # LOG_FILE = './logs/scrapy_output.txt'
)


TICKERS = list(pd.read_csv("./sp500/constituents.csv")
               [['Symbol', 'Security']].itertuples(index=False, name=None))


def etl(response):

    # regex to find the data
    response_text = response.text
    num = re.findall('(?<=div\>\"\,)[0-9\.\"\:\-\, ]*', response_text)
    text = re.findall('(?<=s\: \')\S+(?=\'\, freq)', response_text)

    # convert text to dict via json
    dicts = [json.loads('{'+i+'}') for i in num]
    # create dataframe
    df = pd.DataFrame()
    for ind, val in enumerate(text):
        val = val.replace('-', '_')
        df[val] = dicts[ind].values()
    df.index = dicts[ind].keys()

    return df


class Spider(scrapy.Spider):
    name = "sp500"
    start_urls = []
    custom_settings = CUSTOM_SETTINGS
    # handle_httpstatus_list = [301]

    subsites = [
        "income-statement",
        "balance-sheet",
        "cash-flow-statement",
        "financial-ratios"
    ]

    def start_requests(self) -> Iterable[Request]:
        for ticker, name in TICKERS:
            ticker = ticker.lower()
            name = name.lower().split()[0].replace(".", "").replace(",", "")
            for subsite in self.subsites:
                if not os.path.exists(f'./sp500/{subsite}/{ticker.upper()}.csv'):
                    url = f'https://www.macrotrends.net/stocks/charts/{
                        ticker}/{name}/{subsite}'
                    if QUATERLY_REPORTS:
                        url += "?freq=Q"
                    yield Request(url, dont_filter=True)

    def parse(self, response):
        if response.status == 301:
            new_url = response.headers['Location'].decode()
            if QUATERLY_REPORTS:
                new_url += "?freq=Q"
            print(new_url)
            yield Request(new_url, dont_filter=True)

        if response.status // 100 == 2:
            df = etl(response)
            ticker = response.url.replace(
                "https://www.macrotrends.net/stocks/charts/", "").split("/")[0].upper()
            if "income-statement" in response.url:
                df.to_csv(f'sp500/income-statement/{ticker}.csv')
            if "balance-sheet" in response.url:
                df.to_csv(f'sp500/balance-sheet/{ticker}.csv')
            if "cash-flow-statement" in response.url:
                df.to_csv(f'sp500/cash-flow-statement/{ticker}.csv')
            if "financial-ratios" in response.url:
                df.to_csv(f'sp500/financial-ratios/{ticker}.csv')
