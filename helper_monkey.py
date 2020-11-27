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


# General Functions
def help_():
    helptext = '''Hey, Listen!
    
    USAGE NOTES:
        
        text 1
        
        
        text 2
        
        
        text 3'''
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
    column_name = options.pop('column_name', 'price')
    data = output['data']
    prices = [a['close'] for a in data]
    dates = [a['date'] for a in data]
    price_out = DataFrame(prices, index=dates, columns=[column_name])
    return price_out

def merge_stocks(df_list):
    df_merged = reduce(lambda  left,right: pd.merge(left, right, right_index=True, left_index=True, how='outer'), df_list)
    df_merged = df_merged.dropna()
    return df_merged


"""
Mathematics Functions

Note: Many of the functions in this section have been adapted from the book 
"Machine Learning for Asset Managers" by Marcos M. Lopez de Prado, 
Cambridge University Press, 2020.

"""

# Machine Learning Functions

# covariance matrix
def cov_matrix(df):
    cov = np.cov(df.values.T)
    return cov

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