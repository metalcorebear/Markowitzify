# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:35:44 2020

@author: metalcorebear
"""

import helper_monkey as mojo
import os


"""
This class will construct and analyze a stock portfolio.

"""

class portfolio():
    
    def __init__(self, **options):
        
        self.verbose = options.pop('verbose', False)
        self.API_KEY = options.pop('API_KEY', None)
        self.portfolio = None
        self.cov = None
        self.optimal = None
        self.nco = None
        self.simulation = None
        self.trading_days = 250
        
        if self.verbose:
            print('Verbose is turned on... A long, long time ago the World was in an age of Chaos. In the middle of this chaos, in a little kingdom in the land of Hyrule, a legend was being handed down from generation to generation, the legend of the ''Triforce''; golden triangles possessing mystical powers...')
    
    def help_(self):
        
        mojo.help_()
        
    def about(self):
        
        mojo.about()
        
    def set_API_KEY(self, API_KEY):
        
        if isinstance(API_KEY, str):
            self.API_KEY = API_KEY
            
            if self.verbose:
                print('It''s a secret to everyone.')
        
        else:
            print('Grumble, Grumble')
            print('Input Error: Entry must be a string.')
        
    def set_verbose(self, verbose):
        
        if isinstance(verbose, bool):
            self.verbose = verbose
            
            if self.verbose:
                print('Verbose is turned on... A long, long time ago the World was in an age of Chaos. In the middle of this chaos, in a little kingdom in the land of Hyrule, a legend was being handed down from generation to generation, the legend of the ''Triforce''; golden triangles possessing mystical powers...')
            
            if not self.verbose:
                print('Well excuuuse me, princess!')
                print("Verbose is turned off.  I guess you don't like funny Zelda quotes...")
        
        else:
            print('Grumble, Grumble')
            print('Input Error: Entry must be a bool (True or False).')
    
    def build_TSP(self):
        
        if self.verbose:
            print('It''s dangerous to go alone! Take this.')
            print('Building TSP portfolio...')
        
        self.portfolio = mojo.import_stock_data_DataReader(TSP=True)
        self.optimal = None
        self.simulation = None
        self.cov = None
        
        if self.verbose:
            print('TSP Portfolio complete.')
            print(self.portfolio.head())
        
        if len(self.portfolio.columns) >= 1:
            self.cov = mojo.cov_matrix(self.portfolio)
        
        if self.verbose:
            print('Covariance matrix complete.')
    
    def build_portfolio(self, TKR_list, time_delta, **options):
        
        end_date = options.pop('end_date', None)
        datareader = options.pop('datareader', True)
        
        if not isinstance(TKR_list, list):
            TKR_list = [TKR_list]
        
        date_from, date_to = mojo.get_date_range(time_delta, end_date=end_date)
        
        if datareader:
            
            if self.verbose:
                print('It''s dangerous to go alone! Take this.')
                print('Building portfolio...')
            
            self.portfolio = mojo.import_stock_data_DataReader(start = date_from, tickers=TKR_list)
            
            if self.verbose:
                print('Portfolio complete.')
                print(self.portfolio.head())
                
            if len(self.portfolio.columns) >= 1:
                self.cov = mojo.cov_matrix(self.portfolio)
        
                if self.verbose:
                    print('Covariance matrix complete.')
                    
        else:
        
            df_list = []
            
            if self.verbose:
                print('It''s dangerous to go alone! Take this.')
                print('Building portfolio...')
            
            if self.API_KEY is not None:
    
                for TKR in TKR_list:
                    url = mojo.tradingdata_url(self.API_KEY, TKR, date_to, date_from)
                    
                    try:
                        output, status = mojo.get_json(url)
                        if status == 200:
                            price_out = mojo.get_prices(output, column_name=TKR)
                            df_list.append(price_out)
                        else:
                            print(f'HTTP Error: {status} for {TKR}')
                            
                            if self.verbose:
                                print('Maybe 2020 is just a big 404 error...')
                    
                    except:
                        print(f'Unable to find TKR...skipping {TKR}.')
                
                if len(df_list) >= 1:
                    self.portfolio = mojo.merge_stocks(df_list)
                    
                    if self.verbose:
                        print('Portfolio complete.')
                
                if len(self.portfolio.columns) >= 1:
                        self.cov = mojo.cov_matrix(self.portfolio)
                        
                        if self.verbose:
                            print('Covariance matrix complete.')
            
            else:
                print('Error: Try setting your API Key first...')
    
    def NCO(self, **options):
        
        mu = options.pop('mu', None)
        maxNumClusters = options.pop('max_clusters', 10)
        
        if maxNumClusters >= len(self.portfolio.columns):
            maxNumClusters = len(self.portfolio.columns) - 1
        
        if self.verbose:
            print('Optimizing portfolio...linear algebra is fun!!')
    
        nco = mojo.optPort_nco(self.cov, mu=mu, maxNumClusters=maxNumClusters)
        self.nco = mojo.DataFrame(nco.T, columns=self.portfolio.columns.tolist())

        if self.verbose:
            print('Master using it and you can have this... Portfolio optimized.')
            print(self.nco.head())

    def markowitz(self):
        
        if self.verbose:
            print('Optimizing portfolio...linear algebra is fun!!')
        
        mk = mojo.markowitz(self.portfolio)
        mk = mk.reshape(mk.shape[0], 1)
        #self.optimal = mk
        self.optimal = mojo.DataFrame(mk.T, columns=self.portfolio.columns.tolist())
    
        if self.verbose:
            print('Master using it and you can have this... Portfolio optimized.')
            print(self.optimal.head())
    
    def import_portfolio(self, input_path, **options):
        
        if self.verbose:
            print('It''s dangerous to go alone! Take this.')
            print('Importing portfolio...')
        
        dates_kw = options.pop('dates_kw', 'date')
        filename = options.pop('filename', 'portfolio.csv')
        df = mojo.pd.read_csv(os.path.join(input_path, filename), index_col = 0)
        df.index_name = dates_kw
        
        self.portfolio = df
        
        if self.verbose:
            print('Portfolio complete.')
        
        if len(self.portfolio.columns) >= 1:
                self.cov = mojo.cov_matrix(self.portfolio)
                
                if self.verbose:
                    print('Covariance matrix complete.')

    def sharpe_ratio(self, **options):
        
        if self.optimal is None:
            opt = self.markowitz()
        else:
            opt = self.optimal
        
        w = options.pop('weights', opt)
        risk_free = options.pop('risk_free', 0.035)
        
        if self.verbose:
            print('Calcualting Sharpe ratio...')
            
        self.sharpe = mojo.sharpe(self.portfolio, w, risk_free=risk_free)
        
        if self.verbose:
            print(f'Sharpe Ratio: {self.sharpe}')


    def save(self, file_path, **options):
        
        filename = options.pop('filename', 'portfolio.csv')
        
        full_path = os.path.join(file_path, filename)
        
        try:
            self.portfolio.to_csv(full_path)
            
            if self.verbose:
                print('Portfolio saved.')
        
        except:
            print('I am Error.')
            print('Error: Unable to save portfolio.')
    
    def trend(self, **options):
        
        span = options.pop('span', [3, 10, 1])
        exclude = options.pop('exclude', [])
        output = mojo.DataFrame()
        
        columns = list(self.portfolio.columns)
        
        for item in exclude:
            try:
                columns.remove(item)
            except:
                continue
        
        reduced_portfolio = self.portfolio[columns]
        
        if len(self.portfolio.columns) > 0:
            
            if self.verbose:
                print('Reticulating splines...')
        
            if not isinstance(span, list):
                print('Error: Span must be a list')
                print('Setting span to default: [3, 10, 1]')
                span = [3, 10, 1]
            
            df_list = []
            
            for column in reduced_portfolio.columns:
                out_trend = mojo.GetBinsFromTrend(self.portfolio[column], column, span=span)
                df_list.append(out_trend)
                
            output = mojo.merge_stocks(df_list)
            
        else:
            print('Grumble, Grumble')
            print('Portfolio requires at least one item.')
                
        return output
    
    def simulate(self, threshold=0.2, days=100, **options):
        
        on = options.pop('on', 'return')
        exclude = options.pop('exclude', [])
        iterations = options.pop('iterations', 10000)
        
        columns = list(self.portfolio.columns)
        
        for item in exclude:
            try:
                columns.remove(item)
            except:
                continue
        
        reduced_portfolio = self.portfolio[columns]
        
        probabilities = []
        
        if self.verbose:
            print('Simulating...')
            print('Reticulatiing splines...')
        
        for column in reduced_portfolio.columns:
            out_dict = {'TKR':column}
            sim = mojo.simulate(self.portfolio[column], days, iterations)
            
            prob = mojo.probs_predicted(sim, threshold, on=on)
            breakeven = mojo.probs_predicted(sim, 0.0, on=on)
            roi = mojo.ROI(sim)
            EV = mojo.expected_value(sim)
            
            out_dict.update({'Prob. ' + str(100*threshold) + ' % Return':prob})
            out_dict.update({'Breakeven':breakeven})
            out_dict.update({'ROI':roi})
            out_dict.update({'Expected Value':EV})
            
            probabilities.append(out_dict)
        
        simulation = mojo.DataFrame(probabilities)
        simulation = simulation.set_index('TKR')
        
        if self.verbose:
            print('Simulation complete.')
            print(simulation)
        
        return simulation


    def hurst(self, **options):
        
        if self.verbose:
            print('Calculating Hurst Exponents...')
        
        lag1 = options.pop('lag1', 2)
        lag2 = options.pop('lag2', 20)
        
        H = []
        
        for column in self.portfolio.columns:
            h0 = mojo.hurst(self.portfolio[column], lag1=lag1, lag2=lag2)
            H.append(h0)
        
        H = mojo.np.array(H)
        H = H.reshape(H.shape[0], 1)
        
        self.H = mojo.DataFrame(H.T, columns=self.portfolio.columns.tolist())
        
        if self.verbose:
            print('Hurst Exponents.')
            print(self.H.head())
        

"""
This class will construct and analyze an individual stock or cryptocurrency object.

"""

class stonks():
    
    def __init__(self, TKR, **options):
        
        date_from, _ = mojo.get_date_range(10*360, end_date=None)
        start = options.pop('start', date_from)
        
        self.TKR = TKR
        self.stonk = mojo.import_high_low(start = start, ticker=TKR)
        self.verbose = options.pop('verbose', False)
        self.bands = None
        self.fract = None
        self.rsi = None
        self.sig = None
        self.strategies = None
        self.best_strategy = None
        self.log_stonk = mojo.log_ret(self.stonk)
        
        if self.verbose:
            print('Verbose is turned on... A long, long time ago the World was in an age of Chaos. In the middle of this chaos, in a little kingdom in the land of Hyrule, a legend was being handed down from generation to generation, the legend of the ''Triforce''; golden triangles possessing mystical powers...')
            print('Stonk is {}'.format(self.TKR))
            print(self.stonk.head())
    
    def help_(self):
        
        mojo.help_()
        
    def about(self):
        
        mojo.about()
    
    def set_verbose(self, verbose):
    
        if isinstance(verbose, bool):
            self.verbose = verbose
            
            if self.verbose:
                print('Verbose is turned on... A long, long time ago the World was in an age of Chaos. In the middle of this chaos, in a little kingdom in the land of Hyrule, a legend was being handed down from generation to generation, the legend of the ''Triforce''; golden triangles possessing mystical powers...')
            
            if not self.verbose:
                print('Well excuuuse me, princess!')
                print("Verbose is turned off.  I guess you don't like funny Zelda quotes...")
        
        else:
            print('Grumble, Grumble')
            print('Input Error: Entry must be a bool (True or False).')
    
    def fractal(self, **options):
        
        n = options.pop('n', 20)
        lookback = options.pop('lookback', 14)
        
        if self.verbose:
            print('Calculating Fractal Indicator...')
            print('Retiulating Splines...')
            
        self.fract = mojo.fractal_indicator(self.stonk, n=n, min_max_lookback=lookback)
        
        if self.verbose:
            print(self.fract.head())
        
    def bollinger(self, **options):
        
        n = options.pop('n', 20)
        m = options.pop('m', 2)
        log_ = options.pop('log_returns', True)
        tp, upper, lower = mojo.bollinger(self.stonk, m=m, n=n, log_=log_)
        
        concatenated = mojo.np.concatenate((tp, upper, lower), axis=1)
        
        if self.verbose:
            print('Calculating Bollinger Bands...')
        
        self.bands = mojo.pd.DataFrame(concatenated, columns = ['Trending_Price', 'Upper', 'Lower'])
        
        if self.verbose:
            print(self.bands.head())
    
    def RSI(self, **options):
        
        initial_lookback = options.pop('initial_lookback', 14)
        lookback = options.pop('lookback', 14)
        
        if self.verbose:
            print('Calculating RSI...')        
        
        self.rsi = mojo.RSI(self.stonk, initial_lookback=initial_lookback, lookback=lookback)
        
        if self.verbose:
            print(self.rsi.head())
            
    def signal(self, **options):
        
        lookback = options.pop('lookback', 14)
        
        if self.verbose:
            print('Calculating Bullish/Bearish signal...')        
        
        self.sig = mojo.signal(self.stonk, lookback=lookback)

        if self.verbose:
            print(self.sig.head())        
        
    def strategize(self, **options):
        
        eATR_lookback = options.pop('eATR_lookback', 10)
        buy_range = options.pop('buy_range', (1.0, 4.0, 0.25))
        risk_range = options.pop('risk_range',(1.0, 4.0, 0.25))
        chandelier = options.pop('chandelier', False)
        
        self.eatr = mojo.eATR(self.stonk, lookback=eATR_lookback)
        
        if self.verbose:
            print('Backtesting strategies....')
        
        strategies = mojo.simulate_strategies(self.stonk, buy_range = buy_range, risk_range=risk_range, chandelier=chandelier)
        self.strategies = mojo.pd.DataFrame(strategies)
        
        self.best_strategy = mojo.find_optimal_strategy(strategies)
        
        if self.verbose:
            s = mojo.pd.DataFrame(self.best_strategy, index=[self.TKR])
            print('Optimal Strategy:')
            print(s)