# Markowitzify

(C) 2020 Mark M. Bailey, PhD

## About

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
* Tavora, Marco. "How the Mathematics of Fractals Can Help Predict Stock Markets Shifts," *Medium,* June 2019, Link: https://towardsdatascience.com/how-the-mathematics-of-fractals-can-help-predict-stock-markets-shifts-19fee5dd6574.

## Updates
* 2020-12-17: Added stonks class and methods for individual stock/crypto analysis.
* 2020-12-01: Added Hurst Exponent, Sharpe Ratio, and separated NCO and Markowitz optimization methods.
* 2020-11-28: Added Monte Carlo simulation capability.
* 2020-11-27: Initial commit.

## Installation
`pip install markowitzify`

## Import
`import markowitzify`

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