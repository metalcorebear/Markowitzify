# Markowitzify

(C) 2020 Mark M. Bailey, PhD

## About

Markowitzify will implement portfolio optimization based on the theory described by Harry Markowitz (University of California, San Diego), and elaborated by Marcos M. Lopez de Prado (Cornell University).  In 1952, Harry Markowitz posited that the investment problem can be represented as a convex optimization algorithm.  Markowitz's Critial Line Algorithm (CLA) estimates an "efficient frontier" of portfolios that maximize an expected return based on portfolio risk, where risk is measured as the standard deviation of the returns.  However, solutions to these problems are often mathematically unstable.  Lopez de Prado developed a machine-learning solution called Nested Cluster Optimization (NCO) that addresses this instability.  This repository applies the NCO algorithm to a stock portfolio.

## References
* Lopez de Prado, Marcos M. *Machine Learning for Asset Managers,* Cambridge University Press, 2020.
* Markowitz, Harry. "Portfolio Selection," *Journal of Finance,* Vol. 7, pp. 77-91, 1952.

## Updates
* Initial commit

## Installation
`pip install markowitzify`

## Usage

### Object Instantiation
`import markowitzify`<br>

`portfolio_object = markowitzify.portfolio(**options)`<br><br>

Attributes:<br>
* `portfolio_object.portfolio` = Portfolio (Pandas DataFrame).
* `portfolio_object.cov` = Portfolio covariance matrix (Numpy array).
* `portfolio_object.optimal` = Optimal portfolio configuration (Pandas DataFrame).

Parameters:<br>
* API_KEY (optional) = (str) API Key from Market Stack (only requried if using this method to build portfolio).
* verbose (optional, default = False) = (bool) Turn on if you like Zelda jokes.

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
* TKR_list (required) = (list) List of ticker symbols for portfolio.
* time_delta (required) = (int) Number of days to collect price data (either from today or from end_date).
* end_date (optional, default = today's date) = (str, %m-%d-%Y) Specify the end date for the time delta calculation.<br><br>

Upload CSV:<br>
`portfolio_object.import_portfolio(input_path, **options)`<br>

Parameters:<br>
* input_path (required) = (str) Location of CSV file.
* filename (optional, default = 'portfolio.csv') = (str) Optional file name for portfolio CSV file.
* dates_kw (optional, default = 'date') = (str) Name of column in portfolio that contains the date of each closing price.<br><br>

Export Portfolio:<br>
`portfolio_object.save(file_path, **options)`<br>

Parameters:<br>
* file_path (required) = (str) Location of CSV file.
* filename (optional, default = 'portfolio.csv') = (str) Optional file name for portfolio CSV file.

### Finding Optimal Weights
Implements the Markowitz CLA algorithm.<br><br>

`portfolio_object.optimize(**options)`<br>

Parameters:<br>
* mu (optional, default = None) = (float) When not None, algorithm will return the Sharpe ratio portfolio; otherwise will return the NCO portfolio.
* maxNumClusters (optional, default = 10 or number of stocks in portfolio - 1) = (int) Maximum number of clusters.  Must not exceed the number of stocks in the portfolio - 1.


### Trend Analysis
Trend analysis can be performed on securities within the portfolio.  Output is a Pandas DataFrame.<br><br>

`trend_output = portfolio_object.trend(**options)`<br>

Parameters:<br>
* exclude (optional, default = []) = (list) List of ticker symbols in portfolio to exclude from trend analysis.  Default setting will include all items in portfolio.
