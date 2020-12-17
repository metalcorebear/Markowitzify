# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:21:35 2020

@author: metalcorebear
"""

"""
Helper Monkey contains functions to support all classes.

"""

"""
General Functions

"""
# Import Relevant Libraries

from datetime import date as datemethod
from datetime import timedelta
from datetime import datetime
import requests
from pandas import DataFrame
from pandas import Series
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import pandas as pd
from functools import reduce
import statsmodels.api as sml
from pandas_datareader import data as wb
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
import zipfile, urllib.request, shutil
import os


# General Functions

def about():
     abouttext = '''Hey, Listen!
Markowitzify

(C) 2020 Mark M. Bailey, PhD
MIT License

Markowitzify will implement a variety of portfolio and stock/cryptocurrency analysis methods to optimize portfolios or trading strategies.  The two primary classes are portfolio and stonks.<br>

The portfolio class will implement portfolio optimization based on the theory described by Harry Markowitz (University of California, San Diego), and elaborated by Marcos M. Lopez de Prado (Cornell University).  In 1952, Harry Markowitz posited that the investment problem can be represented as a convex optimization algorithm.  Markowitz's Critial Line Algorithm (CLA) estimates an "efficient frontier" of portfolios that maximize an expected return based on portfolio risk, where risk is measured as the standard deviation of the returns.  However, solutions to these problems are often mathematically unstable.  Lopez de Prado developed a machine-learning solution called Nested Cluster Optimization (NCO) that addresses this instability.  This repository applies both the CLA algorithm, as well as the improved NCO algorithm, to a stock portfolio.  Additionally, this repository simulates portfolio performance over time using Monte Carlo methods, and calculates various other measures of portfolio performance, including the Hurst Exponent and Sharpe Ratio.<br>

The stonks class will create a stock or cryptocurrency object containing OHLC data.  The Relative Strength Indicator (RSI), a Fractal Indicator (as defined by Kaabar), Bollinger Bands, and Bullish/Bearish signals (based on RSI and the Fractal Indicator) can be calculated.  Simulated trading strategies can also be backtested to elucidate an optimal strategy based on maximized profit.

## References
* Carr, Michael. "Measure Volatility With Average True Range," *Investopedia,* Nov 2019, Link: https://www.investopedia.com/articles/trading/08/average-true-range.asp#:~:text=The%20average%20true%20range%20%28ATR%29%20is%20an%20exponential,signals%2C%20while%20shorter%20timeframes%20will%20increase%20trading%20activity.
* Hall, Mary. "Enter Profitable Territory With Average True Range," *Investopedia," Sep 2020, Link: https://www.investopedia.com/articles/trading/08/atr.asp.
* Kaabar, Sofien. "Coding Different Facets of Volatility," *Medium,* Oct 2020, Link: https://medium.com/python-in-plain-english/coding-different-facets-of-volatility-bd1a49282df4.
* Kaabar, Sofien. "Developing a Systematic Indicator to Trade Cryptocurrencies With Python," *Medium,* Dec 2020, Link: https://medium.com/python-in-plain-english/a-technical-indicator-that-works-for-cryptocurrencies-python-trading-258963c7e9c7.
* Kaabar, Sofien. "The Fractal Indicator — Detecting Tops & Bottoms in Markets," *Medium,* Dec 2020, Link: https://medium.com/swlh/the-fractal-indicator-detecting-tops-bottoms-in-markets-1d8aac0269e8.
* Lopez de Prado, Marcos M. *Machine Learning for Asset Managers,* Cambridge University Press, 2020.
* Markowitz, Harry. "Portfolio Selection," *Journal of Finance,* Vol. 7, pp. 77-91, 1952.
* Melul, Elias. "Monte Carlo Simulations for Stock Price Predictions [Python]," *Medium,* May 2018, Link: https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662.
* Tavora, Marco. "How the Mathematics of Fractals Can Help Predict Stock Markets Shifts," *Medium,* June 2019, Link: https://towardsdatascience.com/how-the-mathematics-of-fractals-can-help-predict-stock-markets-shifts-19fee5dd6574.'''
     print(abouttext)
    
def help_():
    helptext = '''Hey, Listen!
    
USAGE NOTES:
        
## The Portfolio Class

### Object Instantiation
`portfolio_object = markowitzify.portfolio(**options)`<br>

Attributes:<br>
* `portfolio_object.portfolio` = Portfolio (Pandas DataFrame).
* `portfolio_object.cov` = Portfolio covariance matrix (Numpy array).
* `portfolio_object.optimal` = Optimal portfolio configuration calculated using the Markowitz (CLA) algorithm (Pandas DataFrame).
* `portfolio_object.nco` = Optimal portfolio configuration calculated using nco algorithm (Pandas DataFrame).
* `portfolio_object.sharpe` = Sharpe ratio for the portfolio (float).
* `portfolio_object.H` = Hurst Exponents for each stock in the portfolio (Pandas DataFrame).
* `portfolio_object.help_()` = View instructions.
* `portfolio_object.about()` = View about.

Parameters:<br>
* `API_KEY` (optional) = (str) API Key from Market Stack (only requried if using this method to build portfolio).
* `verbose` (optional, default = False) = (bool) Turn on if you like Zelda jokes.

### Updating Parameters
Set API Key:<br>
`portfolio_object.set_API_KEY(<STR>)`<br>

Set verbose:<br>
`portfolio_object.set_verbose(<BOOL>)`<br>

### Building Portfolio
Portfolio objects can be instantiated by uploadng a CSV of portfolio performance, or using the Market Stack API (https://marketstack.com/, API key required - note that access may be limited if using an unpaid account).<br><br>

Market Stack API:<br>
`portfolio_object.build_portfolio(TKR_list, time_delta, **options)`<br>

Parameters:<br>
* `datareader` (optional, default = True) = (bool) If True, will use Pandas data_reader to find stock data.  If False, will use Market Stack API (requires API Key).
* `TKR_list` (required) = (list) List of ticker symbols for portfolio.
* `time_delta` (required) = (int) Number of days to collect price data (either from today or from end_date).
* `end_date` (optional, default = today's date) = (str, %m-%d-%Y) Specify the end date for the time delta calculation.<br><br>

Upload CSV:<br>
`portfolio_object.import_portfolio(input_path, **options)`<br>

Parameters:<br>
* `input_path` (required) = (str) Location of CSV file.
* `filename` (optional, default = 'portfolio.csv') = (str) Optional file name for portfolio CSV file.
* `dates_kw` (optional, default = 'date') = (str) Name of column in portfolio that contains the date of each closing price.<br><br>

Build TSP:<br>
Builds a portfolio based on Thrift Savings Plan funds with a lookback of 5 years from the current date.<br>
`portfolio_object.build_TSP()`<br>

Export Portfolio:<br>
`portfolio_object.save(file_path, **options)`<br>

Parameters:<br>
* `file_path` (required) = (str) Location of CSV file.
* `filename` (optional, default = 'portfolio.csv') = (str) Optional file name for portfolio CSV file.

### Finding Optimal Weights
Implements the NCO algorithm.<br>

`portfolio_object.nco(**options)`<br>

Parameters:<br>
* `mu` (optional, default = None) = (float) When not None, algorithm will return the Sharpe ratio portfolio; otherwise will return the NCO portfolio.
* `maxNumClusters` (optional, default = 10 or number of stocks in portfolio - 1) = (int) Maximum number of clusters.  Must not exceed the number of stocks in the portfolio - 1.<br>

Implements the Markowitz optimization algorithm.<br>

`portfolio_object.markowitz()`

### Hurst Exponent and Sharpe Ratios
Calculate the Hurst Exponent for each stock in the portfolio.<br>

`portfolio_object.hurst(**options)`<br>

Parameters:<br>
* lag1, lag2 (optional, default = (2, 20)) = (int) Lag times for fractal calculation.<br>

Calculate the Sharpe ratio for the portfolio.<br>

`H = portfolio_object.sharpe_ratio(**options)`<br>

Parameters:<br>
* w (optional, dafault = Markowitz optimal weights) = (Numpy array) Weights for each stock in the portfolio.
* risk_free (optional, dafault = 0.035) = (float) Risk-free rate of return.

### Trend Analysis
Trend analysis can be performed on securities within the portfolio.  Output is a Pandas DataFrame.<br>

`trend_output = portfolio_object.trend(**options)`<br>

Parameters:<br>
* `exclude` (optional, default = []) = (list) List of ticker symbols in portfolio to exclude from trend analysis.  Default setting will include all items in portfolio.

### Monte Carlo Simulation
Simulated market returns.  Output is a Pandas DataFrame with metrics for all included ticker symbols.<br>

`simulation_output = portfolio_object.simulate(threshold=0.2, days=100, **options)`<br>

Parameters:<br>
* `threshold` (required, dafault = 0.2) = (float) Probability of a 'threshold' return, e.g., 0.2 would calculate the probability of a 20% return.
* `days` (required, default = 100) = (int) Number of days in Monte Carlo simulation.
* `on` (optional, default = 'return') = (str) Predicted return will be calculated on percent return if 'return' or on raw price if 'value'.
* `exclude` (optional, default = []) = (list) List of ticker symbols in portfolio to exclude from trend analysis.  Default setting will include all items in portfolio.
* `iterations` (optional, default = 10000) = (int) Number of iterations in Monte Carlo simulation.

## The Stonks Class

### Object Instantiation
`stock_object = markowitzify.stonks(TKR, **options)`<br>

Attributes:<br>
* `stock_object.TKR` = Ticker symbol (str).
* `stock_object.stonk` = OHLC array (Pandas DataFrame).
* `stock_object.bands` = OHLC with Bollinger Bands (Pandas DataFrame).
* `stock_object.fract` = OHLC with Fractal Indicator (Pandas DataFrame).
* `stock_object.rsi` = OHLC with RSI Indicator (Pandas DataFrame).
* `stock_object.sig` = OHLC with Bullish/Bearish signals based on Fractal Indicator and RSI (-1 == oversold, 1 == overbought) (Pandas DataFrame).
* `stock_object.strategies` = Traading strategies (Buy/Risk coefficients of exponential average true range) and backtesting outcomes (Pandas DataFrame).
* `stock_object.best_strategy` = Optimal strategy that maximizes profit (dictionary).
* `stock_object.help_()` = View instructions.
* `stock_object.about()` = View about.

Parameters:<br>
* `TKR` (required) = (str) Ticker symbol.
* `start` (optional, default = 10 years from today) = (str) Start date to collect OHLC data.
* `verbose` (optional, default = False) = (bool) Turn on if you like Zelda jokes.

### Updating Parameters
Set verbose:<br>
`portfolio_object.set_verbose(<BOOL>)`<br>

### Fractal Indicator
Calculates the Fractal Indicator, as defined by Kaabar (see "The Fractal Indicator — Detecting Tops & Bottoms in Markets").<br>

`stock_object.fractal(**options)`<br>

Parameters:<br>
* `n` (optional, default = 20) = (int) EMA and Rolling Volatility lookback.
* `lookback` (optional, default = 14) = (int) Fractal Indicator lookback.<br>

### Bollinger Bands
Calculates trending price, as well as upper and lower Bollinger Bands.<br>

`stock_object.bollinger(**options)`<br>

Parameters:<br>
* `n` (optional, default = 20) = (int) Number of days in smoothing period.
* `m` (optional, default = 2) = (int) Number of standard deviations.
* `log_returns` (optional, default = True) = (bool) Use Log Returns for calculation of bands.  If False, uses Adjusted Close raw values.<br>

### Relative Strength Indicator
Calculates the RSI for the dataset.<br>

`stock_object.RSI(**options)`<br>

Parameters:<br>
* `initial_lookback` (optional, default = 14) = (int) Lookback period for initial RSI value.
* `lookback` (optional, default = 14) = (int) Lookback period for subsequent RSI values.<br>

### Bullish / Bearish Signal Generator
Determines Bullish or Bearish signal based on Fractal Indicator and RSI signals.  (-1 == oversold, 1 == overbought).<br>

`stock_object.signal(**options)`<br>

Parameters:<br>
* `lookback` (optional, default = 14) = (int) Sets all lookback periods for RSI and Fractal Indicator calculations.<br>

### Simulate Trading Strategies
Simulates and backtests a set of strategies (Buy/Risk coefficients) to find the optimum trading strategy that maximizes profit within a range of "Buy" and "Risk" values. Uses exponential average true range to quantify risk.  "Buy" and "Risk" parameters are multiples of eATR for entry and exit criteria, respectively.  For example, if buy = 1, then the entry criterion is defined as 1x eATR plus the previous day's closing price.  If risk = 2, then a stop loss of 2x the eATR is defined as the exit criterion.<br>

`stock_object.strategize(**options)`<br>

Parameters:<br>
* `eATR_lookback` (optional, default = 10) = (int) Sets lookback periods for exponential average true range.
* `buy_range` (optional, default = (1.0, 4.0, 0.25)) = (tuple) Range of "Buy" coefficients to consider in the model, in the format (start, stop, interval).
* `risk_range` (optional, default = (1.0, 4.0, 0.25)) = (tuple) Range of "Risk" coefficients to consider in the model, in the format (start, stop, interval).
* `chandelier` (optional, default = False) = (bool) If True, uses the chandelier method for determining stop loss (high price - risk*eATR).  If False, uses the closing price instead of high price.<br>
'''
    print(helptext)


# Data Acquisition Functions

def get_date_range(time_delta, end_date=None):
    if end_date is None:
        t = datemethod.today()
    else:
        t = datetime.strptime(end_date, '%m-%d-%Y')
    dt = timedelta(days = time_delta)
    t0 = t - dt
    date_from = datemethod.strftime(t0, '%Y-%m-%d')
    date_to = datemethod.strftime(t, '%Y-%m-%d')
    return date_from, date_to

def tradingdata_url(API_KEY, TKR, date_to, date_from):
    url = f'http://api.marketstack.com/v1/eod?access_key={API_KEY}&symbols={TKR}&date_from={date_from}&date_to={date_to}&sort=asc&limit=1000'
    url = str(url)
    return url

def get_json(url):
    page = requests.request('GET', url)
    status = page.status_code
    output = page.json()
    return output, status

def get_prices(output, **options):
    column_name = options.pop('column_name', 'adj_close')
    data = output['data']
    prices = [a['adj_close'] for a in data]
    dates = [a['date'] for a in data]
    price_out = DataFrame(prices, index=dates, columns=[column_name])
    return price_out

def merge_stocks(df_list):
    df_merged = reduce(lambda  left,right: pd.merge(left, right, right_index=True, left_index=True, how='outer'), df_list)
    df_merged = df_merged.dropna()
    return df_merged

# Pandas Data Reader
def import_stock_data_DataReader(start = '2010-1-1', **options):
    tickers = options.pop('tickers', [])
    TSP = options.pop('TSP', False)
    data = DataFrame()
    if not TSP:
        if len(tickers) != 0:
            if len(tickers) == 1:
                data[tickers] = wb.DataReader(tickers, data_source='yahoo', start=start)['Adj Close']
                data = DataFrame(data)
            else:
                for t in tickers:
                    data[t] = wb.DataReader(t, data_source='yahoo', start=start)['Adj Close']
        else:
            print('Tickers must be specified.')
    else:
        import TSP_Reader
        symbols = ['G Fund', 'F Fund', 'C Fund', 'S Fund', 'I Fund']
        a = TSP_Reader.TSPReader(symbols=symbols)
        data = a.read()
    return data


def import_high_low(start = '2010-1-1', **options):
    ticker = options.pop('ticker', None)
    data = DataFrame()
    if ticker is not None:
        data['Open'] = wb.DataReader(ticker, data_source='yahoo', start=start)['Open']
        data['high'] = wb.DataReader(ticker, data_source='yahoo', start=start)['High']
        data['low'] = wb.DataReader(ticker, data_source='yahoo', start=start)['Low']
        data['Adj Close'] = wb.DataReader(ticker, data_source='yahoo', start=start)['Adj Close']
        data = DataFrame(data)
    else:
        print('Ticker must be specified.')
    return data

"""
Other mathematical functions

"""

# covariance matrix
def cov_matrix(df):
    cov = np.cov(df.values.T)
    return cov

# Log returns
def log_returns(data):
    return (np.log(1 + data.pct_change()))

# Hurst Exponent

def hurst(price_list, lag1=2, lag2=20):
    price_list = price_list.dropna()
    price_list = price_list.values
    lags = range(lag1, lag2)
    tau = [np.sqrt(np.std(np.subtract(price_list[lag:], price_list[:-lag]))) for lag in lags]
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    H = 2*m[0]
    return H

def ret_risk(w, exp_return, cov):
    return -((w.T@exp_return) / (w.T@cov@w)**0.5)

# Markowitz Optimization
def markowitz(df):
    data = log_returns(df)
    data = data.dropna()
    w = np.ones((data.values.T.shape[0],1))*(1.0/data.values.T.shape[0])
    m = np.mean(data.values.T, axis=1)
    demeaned = data.values.T - m[:,None]
    m = m.reshape(m.shape[0],1)
    exp_return = m*w
    cov = np.cov(demeaned)
    opt_bounds = Bounds(0, 1)
    opt_constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})
    res = minimize(ret_risk, w, args = (exp_return, cov), method = 'SLSQP', bounds = opt_bounds, constraints = opt_constraints)
    return res.x
    
# Sharpe Ratio
def sharpe(df, w, risk_free=0.035):
    price_list = log_returns(df)
    price_list = price_list.dropna()
    price_list = price_list.values - np.log(1 + risk_free)
    m = np.mean(price_list.T, axis=1)
    s = np.std(price_list.T, axis=1)
    S_ratio = m/s
    S_ratio = w@S_ratio
    return S_ratio
    

"""
Portfolio Optimization Functions

Note: Many of the functions in this section have been adapted from the book 
"Machine Learning for Asset Managers" by Marcos M. Lopez de Prado, 
Cambridge University Press, 2020.

"""

# Machine Learning Functions

# trend scan
def tValLinR(close):
    # tValue from a linear trend
    x = np.ones((close.shape[0], 2))
    x[:,1] = np.arange(close.shape[0])
    ols = sml.OLS(close, x).fit()
    return ols.tvalues[1]

def GetBinsFromTrend(df, TKR, span=[3, 10, 1]):
    molecule = df.index
    close = df
    out = DataFrame(index=molecule, columns = [TKR+' t1', TKR+' tVal', TKR+' bin'])
    hrzns = range(*span)
    for dt0 in molecule:
        df0 = Series()
        iloc0 = close.index.get_loc(dt0)
        if iloc0 + max(hrzns) > close.shape[0]:
            continue
        for hrzn in hrzns:
            dt1 = close.index[iloc0+hrzn-1]
            df1 = close.loc[dt0:dt1]
            df0.loc[dt1] = tValLinR(df1.values)
        dt1 = df0.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
        out.loc[dt0, [TKR+' t1', TKR+' tVal', TKR+' bin']] = df0.index[-1], df0[dt1], np.sign(df0[dt1])
        #out['t1'] = pd.to_datetime(out['t1'], utc=True)
        out[TKR+' bin'] = pd.to_numeric(out[TKR+' bin'], downcast='signed')
    return out[[TKR+' tVal', TKR+' bin']].dropna(subset=[TKR+' bin'])

# portfolio optimization
def getPCA(matrix):
    # Get eVal, eVec from Hermitian matrix
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:,indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec

def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std, std)
    corr[corr<-1], corr[corr>1] = -1, 1
    return corr

def corr2cov(corr, std):
    cov = corr*np.outer(std, std)
    return cov

def optPort(cov, mu=None):
    # Portfolio optimization function
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu =  ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w

def denoisedCorr(eVal, eVec, nFacts):
    # Remove noise from corr by fixing random eigenvalues
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts] = eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return corr1

def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10):
    corr0 = DataFrame(corr0)
    x, silh = ((1-corr0.fillna(0))/2.)**.5, Series()
    for init in range(n_init):
        for i in range(2, maxNumClusters+1):
            kmeans_ = KMeans(n_clusters=i, n_jobs=1, n_init=1)
            kmeans_ = kmeans_.fit(x)
            kmeans_labels = kmeans_.fit_predict(x)
            silh_ = silhouette_samples(x, kmeans_labels)
            stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0]>stat[1]:
                silh, kmeans = silh_, kmeans_
    newIdx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[newIdx]
    corr1 = corr1.iloc[:, newIdx]
    clstrs = {i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist()\
                              for i in np.unique(kmeans.labels_)}
    silh = Series(silh, index=x.index)
    return corr1, clstrs, silh

def optPort_nco(cov, mu=None, maxNumClusters=10):
    # Portfolio optimizataion function using NCO method
    cov = DataFrame(cov)
    if mu is not None:
        mu = Series(mu[:,0])
    corr1 = cov2corr(cov)
    corr1, clstrs, _ = clusterKMeansBase(corr1, maxNumClusters, n_init=10)
    wIntra = DataFrame(0, index=cov.index, columns=clstrs.keys())
    for i in clstrs:
        cov_ = cov.loc[clstrs[i], clstrs[i]].values
        if mu is None:
            mu_ = None
        else:
            mu_ = mu.loc[clstrs[i]].values.reshape(-1,1)
        wIntra.loc[clstrs[i],i] = optPort(cov_, mu_).flatten()
    cov_ = wIntra.T.dot(np.dot(cov, wIntra))
    mu_ = (None if mu is None else wIntra.T.dot(mu))
    wInter = Series(optPort(cov_, mu_).flatten(), index=cov_.index)
    nco = wIntra.mul(wInter, axis=1).sum(axis=1).values.reshape(-1, 1)
    return nco

"""
Monte Carlo Simulation Function

Note: Functions are adapted from:
Melul, Elias. "Monte Carlo Simulations for Stock Price Predictions [Python]," *Medium,* May 2018, Link: https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662.

"""
# Simulation

def drift_calc(data):
    # Calculate Brownian Motion
    lr = log_returns(data)
    u = lr.mean()
    var = lr.var()
    drift = u-(0.5*var)
    try:
        return drift.values
    except:
        return drift
    
def daily_returns(data, days, iterations):
    ft = drift_calc(data)
    try:
        stv = log_returns(data).std().values()
    except:
        stv = log_returns(data).std()
    dr = np.exp(ft + stv*norm.ppf(np.random.rand(days, iterations)))
    return dr

def probs_predicted(predicted, higherthan, on='value'):
    predicted = DataFrame(predicted)
    if on == 'return':
        predicted0 = predicted.iloc[0,0]
        predicted = predicted.iloc[-1]
        predList = list(predicted)
        over = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 >= higherthan]
        less = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 < higherthan]
    elif on == 'value':
        predicted = predicted.iloc[-1]
        predList = list(predicted)
        over = [i for i in predList if i >= higherthan]
        less = [i for i in predList if i < higherthan]
    else:
        print("'on' must be either 'value' or 'return'")
    return (len(over)/(len(less)+len(over)))

def simulate(data, days, iterations):
    returns = daily_returns(data, days, iterations)
    price_list = np.zeros_like(returns)
    price_list[0] = data.iloc[-1]
    for t in range(1, days):
        price_list[t] = price_list[t-1]*returns[t]
    return price_list

def ROI(price_list):
    price_list_df = DataFrame(price_list)
    out = round((price_list_df.iloc[-1].mean()-price_list[0,1])/price_list_df.iloc[-1].mean(),4)
    return out

def expected_value(price_list):
    price_list_df = DataFrame(price_list)
    out = round(price_list_df.iloc[-1].mean(),2)
    return out

"""
Fractal Indicator Function

Note: Functions are adapted from:
Kaabar, Sofien. "The Fractal Indicator — Detecting Tops & Bottoms in Markets," *Medium,* Dec 2020, Link: https://medium.com/swlh/the-fractal-indicator-detecting-tops-bottoms-in-markets-1d8aac0269e8.
"""
# Simple Moving Average
"""
def sma(x, n=10):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)
"""
def sma(s, n=10):
    out = np.zeros(len(s))
    for i in range(1,len(s)-n):
        out[i+n] = np.mean(s[i:i+n])
    return out

# Exponential Moving Average
def ema(s, n=10):
     ema = np.zeros(len(s))
     multiplier = 2.0 / float(1 + n)
     sma = sum(s[:n]) / float(n)
     ema[n-1] = sma
     for i in range(1,len(s)-n):
         ema[i+n] = s[i+n]*multiplier + ema[i-1+n]*(1-multiplier)
     return ema
     
# Rolling volatility
def rolling_volatility(s, n=10):
    volatility = np.zeros(len(s))
    for i in range(1,len(s)-n):
        volatility[i+n] = np.std(s[i:i+n])
    return volatility

# Fractal Indicator
# df is from import_high_low() function.
def fractal_indicator(df, n=20, min_max_lookback=14):
    np.seterr(divide='ignore', invalid='ignore')
    
    high = df['high'].values
    low = df['low'].values
    
    ema_high = ema(high, n=n)
    ema_low = ema(low, n=n)
    
    vol_high = rolling_volatility(high, n=n)
    vol_low = rolling_volatility(low, n=n)
    ave_vol = (vol_high + vol_low)/2.0
    
    demeaned_high = high - ema_high
    demeaned_low = low - ema_low
    
    max_high = np.zeros(len(df))
    min_low = np.zeros(len(df))
    
    for i in range(min_max_lookback, len(df)):
        max_high[i] = max(demeaned_high[(i - min_max_lookback + 1):(i + 1)])
        min_low[i] = min(demeaned_low[(i - min_max_lookback + 1):(i + 1)])
        
    fractal_1 = max_high-min_low
    
    fractal = np.divide(fractal_1, ave_vol)
    
    out_df = DataFrame(fractal, columns=['Fractal_Indicator'])
    out_df = out_df.replace([np.nan, np.inf], 0)
    out_df = out_df.set_index(df.index)
    
    df_out = df.merge(out_df, right_index=True, left_index=True)
    
    return df_out
    

"""
Market Sentiment Data

Note: Functions are adapted from:
Kaabar, Sofien. "Using Python to Download Sentiment Data for Financial Trading," *Medium,* Nov 2020, Link: https://medium.com/swlh/using-python-to-download-sentiment-data-for-financial-trading-1c44346926e2.
"""

# Download Committment of Traders Report

def get_COT(url, file_name):
    if not os.path.exists(r'COT'):
        os.mkdir(r'COT')
    file_path = os.path.join(r'COT', file_name)
    with urllib.request.urlopen(url) as response, open(file_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(path=r'COT')
    year = file_name.split('.')[0]
    new_file_name = year + '.xls'
    os.rename(r'COT/FinFutYY.xls',os.path.join(r'COT', new_file_name))
    
def get_COT_df(COT_path):
    data = pd.read_excel(COT_path)
    data = data[['Market_and_Exchange_Names', 
                                 'Report_Date_as_MM_DD_YYYY',
                                 'Pct_of_OI_Dealer_Long_All',
                                 'Pct_of_OI_Dealer_Short_All',
                                 'Pct_of_OI_Lev_Money_Long_All',
                                 'Pct_of_OI_Lev_Money_Short_All']]
    data['net_hedgers'] = data['Pct_of_OI_Dealer_Long_All'] - data['Pct_of_OI_Dealer_Short_All']
    data['net_funds'] = data['Pct_of_OI_Lev_Money_Long_All'] - data['Pct_of_OI_Lev_Money_Short_All']
    data = data[['Market_and_Exchange_Names','Report_Date_as_MM_DD_YYYY','net_hedgers','net_funds']]
    return data

# Get and concatenate all COTs data for 2010 through current year.
    
def build_COT():
    url_base = r'https://www.cftc.gov/files/dea/history/fut_fin_xls_'
    t = datemethod.today()
    end_year = t.year + 1
    year_list = list(range(2010, end_year))
    year_list = [str(i) + '.zip' for i in year_list]
    year_list = [i for i in year_list if not os.path.exists(os.path.join('COT', i))]
    if len(year_list) != 0:
        url_list = [url_base + i for i in year_list]
        for i in range(len(year_list)):
            try:
                get_COT(url_list[i], year_list[i])
            except:
                continue
    xls_list = []
    for i in os.walk(r'COT'):
        out_tuple = i
    xls_list = [i for i in out_tuple[2] if '.xls' in i]
    xls_list = [os.path.join('COT', i) for i in xls_list]
    df_list = []
    for i in xls_list:
        try:
            df = get_COT_df(i)
            df_list.append(df)
        except:
            continue
    out_df = pd.concat(df_list)
    out_df = out_df.drop_duplicates()
    out_df = out_df.reset_index()
    out_df = out_df[['Market_and_Exchange_Names','Report_Date_as_MM_DD_YYYY','net_hedgers','net_funds']]
    return out_df
    
    
# Bollinger Bands
# Data are from import_high_low() function.
def bollinger(data, m=2, n=20, log_=True):
    tp = (data['high'].values + data['low'].values + data['Adj Close'].values)/3.0
    ma = sma(tp, n=n)
    sd = rolling_volatility(tp, n=n)
    upper = ma[:] + m*sd[:]
    lower = ma[:] - m*sd[:]
    if log_:
        upper = log_returns(pd.DataFrame(upper))
        tp = log_returns(pd.DataFrame(tp))
        lower = log_returns(pd.DataFrame(lower))
        upper = upper.replace([np.nan, np.inf], 0)
        tp = tp.replace([np.nan, np.inf], 0)
        lower = lower.replace([np.nan, np.inf], 0)
        upper = upper.values
        tp = tp.values
        lower = lower.values
    return tp, upper, lower


# Relative Strength Index
# EMA Method
def RSI(data1, initial_lookback=14, lookback=14):
    data2 = data1['Adj Close']
    data2 = data2.pct_change()
    data2 = data2.replace(np.nan, 0)
    data = data2.values
    pos = [abs(i) if i > 0 else 0.0 for i in data[0:initial_lookback]]
    neg = [abs(i)  if i < 0 else 0.0 for i in data[0:initial_lookback]]
    pos = np.array(pos)
    neg = np.array(neg)
    ave_gain = np.average(pos)
    ave_loss = np.average(neg)
    rsi1 = 100.0 - (100.0 / (1 + ave_gain/ave_loss))
    rsi_out = np.zeros(len(data))
    rsi_out[initial_lookback] = rsi1
    alpha = 2.0/(lookback+1.0)
    for i in range(initial_lookback+1, len(data)):
        pos = [abs(j) if j > 0 else 0.0 for j in data[i-lookback:i]]
        neg = [abs(j) if j < 0 else 0.0 for j in data[i-lookback:i]]
        pos = np.array(pos)
        neg = np.array(neg)
        ave_gain = (1.0-alpha)*ave_gain + alpha*np.average(pos)
        ave_loss = (1.0-alpha)*ave_loss + alpha*np.average(neg)
        rsi1 = 100.0 - (100.0 / (1 + ave_gain/ave_loss))
        rsi_out[i] = rsi1
    
    out_df = DataFrame(rsi_out, columns=['RSI'])
    out_df = out_df.replace([np.nan, np.inf], 0)
    out_df = out_df.set_index(data1.index)
    
    df_out = data1.merge(out_df, right_index=True, left_index=True)
    return df_out


# Bullish/Bearish Signal
def signal(data, lookback=14):
    rsi_out = RSI(data, initial_lookback=lookback, lookback=lookback)
    fractal = fractal_indicator(data, n=20, min_max_lookback=lookback)
    rsi = rsi_out['RSI']
    frac = fractal['Fractal_Indicator']
    data = data.merge(rsi, right_index=True, left_index=True)
    data = data.merge(frac, right_index=True, left_index=True)
    data = data[data['Fractal_Indicator'] != 0]
    data = data[data['RSI'] != 0]
    rsi_ = [-1.0 if data['RSI'][i] <= 30 else (1.0 if data['RSI'][i] >= 70 else 0.0) for i in range(len(data))]
    rsi_ = np.array(rsi_)
    rsi_ = pd.DataFrame(rsi_, columns=['RSI_Signal'])
    rsi_ = rsi_.set_index(data.index)
    #data = data.assign(RSI_signal = lambda x: (-1 if x['RSI'].values <= 30 else (1 if x['RSI'].values >= 70 else 0)))
    trend = [data['Adj Close'][i] - data['Adj Close'][i-lookback] if (i-lookback) > 0 else 0 for i in range(len(data))]
    fr = [1.0 if data['Fractal_Indicator'][i] >= 1 else 0 for i in range(len(data))]
    trend1 = [-1.0 if trend[i] < 0 else (1.0 if trend[i] > 0 else 0) for i in range(len(trend))]
    fr_trend1 = [fr[i]*trend1[i] for i in range(len(fr))]
    fr_trend1 = np.array(fr_trend1)
    fr_trend1 = pd.DataFrame(fr_trend1, columns=['Fractal_Signal'])
    fr_trend1 = fr_trend1.set_index(data.index)
    sig_out = data.merge(fr_trend1, right_index=True, left_index=True)
    sig_out = sig_out.merge(rsi_, right_index=True, left_index=True)
    return sig_out
    
    
# Calculate log returns
def log_ret(data):
    high = data['high']
    low = data['low']
    adj_close = data['Adj Close']
    
    high = log_returns(high)
    low = log_returns(low)
    adj_close = log_returns(adj_close)
    
    high = high.replace([np.nan, np.inf], 0)
    low = low.replace([np.nan, np.inf], 0)
    adj_close = adj_close.replace([np.nan, np.inf], 0)
    
    out_df = pd.concat([high, low, adj_close], axis=1)
    
    return out_df
    
# Stock trade strategy simulation

# Exponential average true range
def eATR(data, lookback=10):
    m = data.values
    z = np.zeros((m.shape[0], 2))
    m = np.concatenate((m, z), axis=1)
    columns = ['Open', 'high', 'low', 'Adj Close', 'ATR', 'eATR']
    # calculate ATR values
    for i in range(1, len(m)):
        atr = [m[i,1] - m[i,2], abs(m[i,1] - m[i-1,3]), abs(m[i-1,3] - m[i,2])]
        m[i,4] = max(atr)
    # calcualate exponential moving average
    alpha = 2.0/float(lookback+1.0)
    sma = sum(m[:lookback,4]) / float(lookback)
    m[lookback,5] = sma
    for i in range(1,len(m)-lookback):
         m[i+lookback,5] = m[i+lookback,4]*alpha + m[i-1+lookback,5]*(1.0-alpha)
    out = pd.DataFrame(m, columns=columns, index=data.index)
    return out


def strategize(data, strategy={'buy':1.0, 'risk':1.0}, chandelier=False):
    m = data.values
    z = np.zeros((m.shape[0], 2))
    m = np.concatenate((m, z), axis=1)
    columns = ['Open', 'high', 'low', 'Adj Close', 'ATR', 'eATR', 'buy_point', 'sell_point']
    for i in range(1, len(m)):
        if (m[i,3] > (m[i-1,3] + strategy['buy']*m[i-1,5])) and (m[i-1,5]>0):
            m[i,6] = 1
        if chandelier:
            if (m[i,3] < (m[i-1,1] - strategy['risk']*m[i-1,5])) and (m[i-1,5]>0):
                m[i,7] = 1
        else:
            if (m[i,3] < (m[i-1,3] - strategy['risk']*m[i-1,5])) and (m[i-1,5]>0):
                m[i,7] = 1
    out = pd.DataFrame(m, columns=columns, index=data.index)
    return out


def evaluate(data, risk_factor=1.0, chandelier=False):
    m = data.values
    profits = []
    risk_rewards = []
    winning = 0
    all_trades = 0
    
    j = 0
    j_start = 1
    
    for i in range(1,len(m)-1):
        
        if m[i,6] == 1:
            buy = m[i,3]
            if j_start < i:
                j_start = i+1
            else:
                j_start = j
            for j in range(j_start,len(m)):
                if m[j,7] == 1:
                    sell = m[j,3]
                    all_trades += 1
                    profit = sell-buy
                    profits.append(profit)
                    if chandelier:
                        stop = m[i-1,1] - risk_factor*m[i,5] 
                    else:
                        stop = m[i-1,3] - risk_factor*m[i,5]
                    risk_reward = profit / (buy - stop)
                    risk_rewards.append(risk_reward)
                    if profit > 0:
                        winning += 1
                    break
    
    if len(profits) != 0:   
        expected = np.average(np.array(profits))
    else:
        expected = 0.0
    total_profit = sum(profits)
    if all_trades != 0:
        hit_ratio = float(winning)/float(all_trades)
    else:
        hit_ratio = 0.0
    gross_profits = [k for k in profits if k > 0]
    gross_losses = [abs(k) for k in profits if k < 0]
    if sum(gross_losses) != 0.0:
        profit_factor = sum(gross_profits) / sum(gross_losses)
    else:
        profit_factor = np.nan
    if len(risk_rewards) != 0:
        risk_reward = np.average(np.array(risk_rewards))
    else:
        risk_reward = 0.0
    output = {'hit_ratio':round(hit_ratio,2), 'total_trades':all_trades, 'expected':round(expected,2), 'total_profit':round(total_profit,2), 'profit_factor':round(profit_factor,2), 'risk_ratio':round(risk_reward,2)}
    return output
    

# Find optimal strategy
def simulate_strategies(data, buy_range = (1.0, 4.0, 0.25), risk_range=(1.0, 4.0, 0.25), chandelier=False):
    buy_i = (buy_range[1] - buy_range[0])/buy_range[2]
    buy_test = [buy_range[0] + float(i)*buy_range[2] for i in range(int(buy_i)+1)]
    risk_i = (risk_range[1] - risk_range[0])/risk_range[2]
    risk_test = [risk_range[0] + float(i)*risk_range[2] for i in range(int(risk_i)+1)]
    strategies = []
    for i in buy_test:
        for j in risk_test:
            s = {'buy':i, 'risk':j}
            strategies.append(s)
    
    for strategy in strategies:
        eatr = eATR(data)
        strat = strategize(eatr, strategy=strategy, chandelier=chandelier)
        sim = evaluate(strat, risk_factor=strategy['risk'], chandelier=chandelier)
        strategy.update(sim)
    return strategies

def find_optimal_strategy(strategies):
    previous_number_to_beat = 0.0
    best_strategy = None
    for strategy in strategies:
        profit = strategy['total_profit']
        if profit > previous_number_to_beat:
            best_strategy = strategy
            previous_number_to_beat = profit
    return best_strategy