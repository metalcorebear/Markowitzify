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


# General Functions

def about():
     abouttext = '''Hey, Listen!
Markowitzify

(C) 2020 Mark M. Bailey, PhD
MIT License

Markowitzify will implement portfolio optimization based on the theory described by Harry Markowitz (University of California, San Diego), and elaborated by Marcos M. Lopez de Prado (Cornell University).  In 1952, Harry Markowitz posited that the investment problem can be represented as a convex optimization algorithm.  Markowitz's Critial Line Algorithm (CLA) estimates an "efficient frontier" of portfolios that maximize an expected return based on portfolio risk, where risk is measured as the standard deviation of the returns.  However, solutions to these problems are often mathematically unstable.  Lopez de Prado developed a machine-learning solution called Nested Cluster Optimization (NCO) that addresses this instability.  This repository applies the NCO algorithm to a stock portfolio.  Additionally, this repository simulates individual stock performance over time using Monte Carlo methods, and performs various other calculations.

## References
* Lopez de Prado, Marcos M. *Machine Learning for Asset Managers,* Cambridge University Press, 2020.
* Markowitz, Harry. "Portfolio Selection," *Journal of Finance,* Vol. 7, pp. 77-91, 1952.
* Melul, Elias. "Monte Carlo Simulations for Stock Price Predictions [Python]," *Medium,* May 2018, Link: https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662.
* Tavora, Marco. "How the Mathematics of Fractals Can Help Predict Stock Markets Shifts," *Medium,* June 2019, Link: https://towardsdatascience.com/how-the-mathematics-of-fractals-can-help-predict-stock-markets-shifts-19fee5dd6574.
     '''
     print(abouttext)
    
def help_():
    helptext = '''Hey, Listen!
    
USAGE NOTES:
        
## Usage

### Object Instantiation
`import markowitzify`<br>

`portfolio_object = markowitzify.portfolio(**options)`<br><br>

Attributes:<br>
* `portfolio_object.portfolio` = Portfolio (Pandas DataFrame).
* `portfolio_object.cov` = Portfolio covariance matrix (Numpy array).
* `portfolio_object.optimal` = Optimal portfolio configuration calculated using the Markowitz CLA algorithm (Pandas DataFrame).
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
`portfolio_object.set_API_KEY(<STR>)`<br><br>

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
Implements the NCO CLA algorithm.<br><br>

`portfolio_object.nco(**options)`<br>

Parameters:<br>
* `mu` (optional, default = None) = (float) When not None, algorithm will return the Sharpe ratio portfolio; otherwise will return the NCO portfolio.
* `maxNumClusters` (optional, default = 10 or number of stocks in portfolio - 1) = (int) Maximum number of clusters.  Must not exceed the number of stocks in the portfolio - 1.<br><br>

Implements the Markowitz optimization algorithm.<br><br>

`portfolio_object.optimize()`

### Hurst Exponent and Sharpe Ratios
Calculate the Hurst Exponent for each stock in the portfolio.<br><br>

`portfolio_object.hurst(**options)`<br>

Parameters:<br>
* lag1, lag2 (optional, dafault = (2, 20)) = (int) Lag times for fractal calculation.<br><br>

Calculate the Sharpe ratio for the portfolio.<br><br>

`H = portfolio_object.sharpe_ratio(**options)`<br>

Parameters:<br>
* w (optional, dafault = Markowitz optimal weights) = (Numpy array) Weights for each stock in the portfolio.
* risk_free (optional, dafault = 0.035) = (float) Risk-free rate of return.

### Trend Analysis
Trend analysis can be performed on securities within the portfolio.  Output is a Pandas DataFrame.<br><br>

`trend_output = portfolio_object.trend(**options)`<br>

Parameters:<br>
* `exclude` (optional, default = []) = (list) List of ticker symbols in portfolio to exclude from trend analysis.  Default setting will include all items in portfolio.

### Monte Carlo Simulation
Simulated market returns.  Output is a Pandas DataFrame with metrics for all included ticker symbols.<br><br>

`simulation_output = portfolio_object.simulate(threshold=0.2, days=100, **options)`<br>

Parameters:<br>
* `threshold` (required, dafault = 0.2) = (float) Probability of a 'threshold' return, e.g., 0.2 would calculate the probability of a 20% return.
* `days` (required, default = 100) = (int) Number of days in Monte Carlo simulation.
* `on` (optional, default = 'return') = (str) Predicted return will be calculated on percent return if 'return' or on raw price if 'value'.
* `exclude` (optional, default = []) = (list) List of ticker symbols in portfolio to exclude from trend analysis.  Default setting will include all items in portfolio.
* `iterations` (optional, default = 10000) = (int) Number of iterations in Monte Carlo simulation.
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