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
        self.simulation = None
        
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
                print('Verbose is turned off.  I guess you don''t like funny Zelda quotes...')
        
        else:
            print('Grumble, Grumble')
            print('Input Error: Entry must be a bool (True or False).')
        
    def build_portfolio(self, TKR_list, time_delta, **options):
        
        end_date = options.pop('end_date', None)
        datareader = options.pop('datareader', True)
        
        if not isinstance(TKR_list, list):
            TKR_list = [TKR_list]
        
        date_from, date_to = mojo.get_date_range(time_delta, end_date=end_date)
        
        if datareader:
            
            self.portfolio = mojo.import_stock_data_DataReader(TKR_list, start = date_from)
            
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


    def optimize(self, **options):
        
        mu = options.pop('mu', None)
        maxNumClusters = options.pop('max_clusters', 10)
        
        if maxNumClusters >= len(self.portfolio.columns):
            maxNumClusters = len(self.portfolio.columns) - 1
        
        if self.verbose:
            print('Optimizing portfolio...linear algebra is fun!!')
    
        nco = mojo.optPort_nco(self.cov, mu=mu, maxNumClusters=maxNumClusters)
        self.optimal = mojo.DataFrame(nco.T, columns=self.portfolio.columns.tolist())

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
            