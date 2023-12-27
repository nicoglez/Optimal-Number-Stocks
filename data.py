import pandas as pd
import numpy as np
import yfinance as yf


tickers=list(pd.read_excel("files/tickers.xlsx")["tickers"])
random_tickers=list(np.random.choice(tickers, size=70, replace=False))
data=yf.download(random_tickers, start="2020-01-01", end="2023-12-31")["Adj Close"]
data_benchmark=yf.download("SPY", start="2020-01-01", end="2023-12-31")["Adj Close"]