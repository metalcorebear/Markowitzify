# Markowitzify

(C) 2020 Mark M. Bailey, PhD

## About

Markowitzify will implement portfolio optimization based on the theory described by Harry Markowitz (University of California, San Diego), and elaborated by Marcos M. Lopez de Prado (Cornell University).  In 1952, Harry Markowitz posited that the investment problem can be represented as a convex optimization algorithm.  Markowitz's Critial Line Algorithm (CLA) estimates an "efficient frontier" of portfolios that maximize an expected return based on portfolio risk, where risk is measured as the standard deviation of the returns.  However, solutions to these problems are often mathematically unstable.  Lopez de Prado developed a machine-learning solution called Nested Cluster Optimization (NCO) that addresses this instability.  This repository applies the NCO algorithm to a stock portfolio.  Additionally, this repository simulates individual stock performance over time using Monte Carlo methods, and performs various other calculations.

## References
* Lopez de Prado, Marcos M. *Machine Learning for Asset Managers,* Cambridge University Press, 2020.
* Markowitz, Harry. "Portfolio Selection," *Journal of Finance,* Vol. 7, pp. 77-91, 1952.
* Melul, Elias. "Monte Carlo Simulations for Stock Price Predictions [Python]," *Medium,* May 2018, Link: https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662.
* Tavora, Marco. "How the Mathematics of Fractals Can Help Predict Stock Markets Shifts," *Medium,* June 2019, Link: https://towardsdatascience.com/how-the-mathematics-of-fractals-can-help-predict-stock-markets-shifts-19fee5dd6574.

## Updates
* 2020-12-01: Added Hurst Exponent, Sharpe Ratio, and separated NCO and Markowitz optimization methods.
* 2020-11-28: Added Monte Carlo Simulation capability.
* 2020-11-27: Initial commit.

## Installation
`pip install markowitzify`

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

`portfolio_object.markowitz()`

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