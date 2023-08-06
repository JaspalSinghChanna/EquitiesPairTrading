from data_loader import MongoDBConnection
import pandas as pd
import datetime
import numpy as np
import logging
from sklearn import linear_model
import statsmodels.api as sm
import pykalman
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
from scipy.stats import zscore

class Analytics:
    def __init__(self):
        self.Mongo = MongoDBConnection()

    def get_all_close_data(self):
        logging.info(datetime.datetime.now())
        cursor = self.Mongo.closecoll.find()
        l1 = []
        for record in cursor:
            l1.append(record)
        logging.info(datetime.datetime.now())
        df2 = pd.DataFrame.from_dict(l1).drop(columns=['_id'])
        df2['Date'] = pd.to_datetime(df2['Date'])
        df2 = df2.set_index('Date')
        return df2

    def get_pairs(self, df2): # only positive correlations
        corr = pd.DataFrame(data=np.corrcoef(df2.values, rowvar=False), index=df2.columns, columns=df2.columns)
        correlation_matrix_upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        correlation_pairs = correlation_matrix_upper.unstack().sort_values(ascending=False).dropna()
        pairs = correlation_pairs.to_frame().reset_index()
        pairs.columns = ['Security 1','Security 2','Correlation']
        return pairs.round(5)

    def get_mean_reversion_speed(self, df_close):
        df_lag = df_close.shift(1)
        df_delta = df_close - df_lag
        lin_reg_model = linear_model.LinearRegression()
        lin_reg_model.fit(df_lag[1:], df_delta[1:])       
        half_life = -np.log(2) / lin_reg_model.coef_.item()
        return half_life

    def fit_OLS(self, price_df, y, x):
        y = price_df[y]
        x = price_df[x]
        x = sm.add_constant(x)
        model = sm.OLS(endog=y, exog=x)
        res = model.fit()
        const = res.params[0]
        beta = res.params[1]
        price_df['Residuals'] = res.resid
        return price_df, beta, const

    def run_Kalman_filter(self, df, y, x):
        orig_y = y
        orig_x = x
        F = np.eye(2)
        x = df[x].to_list()
        n = len(x)
        H = np.vstack([np.matrix(x), np.ones((1, n))]).T[:, np.newaxis]
        # transition_covariance 
        Q = [[1e-4,     0], 
            [   0,  1e-4]] 
        R = np.var(df['Residuals'])
        # initial_state_mean
        X0 = [0,
            0]
        # initial_state_covariance
        P0 = [[  1,    0], 
            [  0,    1]]
        #return [F,H,Q,R,X0,P0]
        # Kalman-Filter initialization
        kf = pykalman.KalmanFilter(n_dim_obs=1, n_dim_state=2,
                        transition_matrices = F, 
                        observation_matrices = H, 
                        transition_covariance = Q, 
                        observation_covariance = R, 
                        initial_state_mean = X0, 
                        initial_state_covariance = P0)
        
        y = df[y].to_list()
        state_means, state_covs = kf.filter(y)
        y_hat = np.multiply(x, state_means[:, 0]) + state_means[:, 1]
        residuals = y - y_hat
        return pd.DataFrame({orig_y: y, 'Pred': y_hat, 'Beta':state_means[:, 0], 
                            'Alpha':state_means[:, 1], orig_x: x, 'Residuals':residuals}, index=df.index)

    def hurst(self, ts):
        # Create the range of lag values
        lags = range(2, 100)
        # Calculate the array of the variances of the lagged differences
        tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        # Use a linear fit to estimate the Hurst Exponent
        poly = polyfit(log(lags), log(tau), 1)
        # Return the Hurst exponent from the polyfit output
        return poly[0]*2.0

    def get_ohlc_data(self, ticker):
        cursor = self.Mongo.ohlccoll.find({'Ticker':ticker})
        l1 = []
        for record in cursor:
            l1.append(record)
        df = pd.DataFrame(l1).set_index('Date').drop(columns=['_id','Ticker','Volume']).dropna()
        return df

    def create_backtest_input_OLS(self, y, x, constant, weighting, start_date=None, end_date=None):
        _y = self.get_ohlc_data(y)
        _x = self.get_ohlc_data(x)
        if start_date:
            _y = _y[start_date:]
            _x = _x[start_date:]
        if end_date:
            _y = _y[:end_date]
            _x = _x[:end_date]
        prices_df = _y - _x
        prices_df['spread'] = _y['Close'] - constant - weighting*_x['Close']
        prices_df['zscore'] = zscore(prices_df['spread'])
        return prices_df

    def create_backtest_input_Kalman(self, y, x, df, start_date=None, end_date=None):
        _y = self.get_ohlc_data(y)
        _x = self.get_ohlc_data(x)
        if start_date:
            _y = _y[start_date:]
            _x = _x[start_date:]
        if end_date:
            _y = _y[:end_date]
            _x = _x[:end_date]
        prices_df = _y - _x
        prices_df['spread'] = _y['Close'] - _x['Close']*df['Beta'] - df['Alpha']
        prices_df.dropna(inplace=True)
        prices_df['zscore'] = zscore(prices_df['spread'])
        return prices_df

    def sharpe_ratio(self, returns, annual_risk_free_rate):
        average_return = np.mean(returns)
        standard_deviation = np.std(returns)
        sharpe_ratio = (average_return - (annual_risk_free_rate
                        /252)) / standard_deviation
        return round(sharpe_ratio,5)

    def sortino_ratio(self, returns, annual_risk_free_rate):
        average_return = np.mean(returns)
        standard_deviation = np.std(returns[returns<0])
        sortino_ratio = (average_return - (annual_risk_free_rate
                        /252)) / standard_deviation
        return round(sortino_ratio,5)

    def _close_out_positions(self, position, gdf, i, current_trade={}):
        #print('Position:' + str(position))
        if position > 0: #the want to sell to close out position
            current_trade['price'] = gdf.iloc[i+1].Open
            current_trade['signal_date'] = gdf.iloc[i].name
            current_trade['trade_date'] = gdf.iloc[i+1].name
            current_trade['volume'] = - position # TODO COME BACK TO THIS
            current_trade['cashflow'] = - current_trade['price'] * current_trade['volume'] #vol is negative so need to add a neg hre
            current_trade['type'] = 'SELL'

        elif position < 0:
            current_trade['price'] = gdf.iloc[i+1].Open
            current_trade['signal_date'] = gdf.iloc[i].name
            current_trade['trade_date'] = gdf.iloc[i+1].name
            current_trade['volume'] = - position # TODO COME BACK TO THIS
            current_trade['cashflow'] = - current_trade['price'] * current_trade['volume']
            current_trade['type'] = 'BUY'
        return current_trade

    def backtest(self, gdf, zscore_sell_threshhold, zscore_buy_threshhold,
                trade_size, position_limit, max_loss, starting_capital):
        trades = []
        stop_trading = False
        position = 0
        #opv = []
        for i in range(len(gdf) - 1):
            current_trade = {}
            z = gdf.iloc[i].zscore
            open_position_value = position * gdf.iloc[i].Close
            cum_cashflow = sum(trade['cashflow'] for trade in trades)
            if cum_cashflow + open_position_value <= - starting_capital:
                logging.warning('Lost all capital (open_position_value:'
                            f'{open_position_value}, cflow: {cum_cashflow})')
                current_trade = self._close_out_positions(position, gdf, i, current_trade={})
                trades.append(current_trade)
                position = 0
                break
            #opv.append(open_position_value)
            if stop_trading==True:
                if z > -0.5 and z < 0.5:
                    logging.warning('Trading resumed')
                    stop_trading = False
            elif abs(open_position_value) > max_loss: #CLOSE OUT POSITION
                stop_trading = True
                logging.warning('Trading stopped')
                current_trade = self._close_out_positions(position, gdf, i, current_trade={})
                trades.append(current_trade)
                position = 0
                
            elif z < zscore_buy_threshhold and position + trade_size <= position_limit: # is signal says buy and have room to buy
                current_trade['price'] = gdf.iloc[i+1].Open
                current_trade['signal_date'] = gdf.iloc[i].name
                current_trade['trade_date'] = gdf.iloc[i+1].name
                current_trade['volume'] = trade_size # TODO COME BACK TO THIS
                current_trade['cashflow'] = - current_trade['price'] * current_trade['volume']
                current_trade['type'] = 'BUY'
                trades.append(current_trade)
                position += trade_size
            elif z > zscore_sell_threshhold and position - trade_size >= - position_limit:
                current_trade['price'] = gdf.iloc[i+1].Open
                current_trade['signal_date'] = gdf.iloc[i].name
                current_trade['trade_date'] = gdf.iloc[i+1].name
                current_trade['volume'] = - trade_size # TODO COME BACK TO THIS
                current_trade['cashflow'] = - current_trade['price'] * current_trade['volume'] #vol is negative so need to add a neg hre
                current_trade['type'] = 'SELL'
                trades.append(current_trade)
                position -= trade_size
        #if trades == []:
            #t1 = pd.DataFrame(columns=['price','signal_date','trade_date','volume','cashflow','type'])
        #else:
        t1 = pd.DataFrame(trades)
        try:
            t1['vol_cum'] = t1['volume'].cumsum()
        except Exception as e:
            logging.warning('Caught problem!')
            raise Exception(e, trades) from e
        t1['cashflow_cum'] = t1['cashflow'].cumsum()
        t2 = t1.set_index('trade_date').join(gdf, how='right')
        t2['vol_cum2'] = t2['vol_cum'].ffill().fillna(0)
        t2['position_value'] = t2['vol_cum2'] * t2['Close']
        t2['cashflow_cum2'] = t2['cashflow_cum'].ffill().fillna(0)
        t2['position_plus_winnings'] = t2['cashflow_cum2'] + t2['position_value']
        t2['returns_%'] = t2['position_plus_winnings'].diff() * 100 / starting_capital
        t2.iloc[0,-1] = 0
        t2['cummax'] = (t2['position_plus_winnings'] + 
                        starting_capital).cummax()
        t2['drawdown'] = (t2['position_plus_winnings'] 
                        + starting_capital - t2['cummax'])/t2['cummax']
        return t2
    
    def get_optimised_params(self, exp_df1, cap, rf):
        #print(datetime.datetime.now())
        _results = {} # relies on starting capital, exp_df1 and rf
        for _z in [1,2]:
            for _ts in [100, 1_000]:
                for _pl in [2_000, 10_000]:
                    for _ml in [0.05, 0.1]:
                        _trial = self.backtest(exp_df1, zscore_sell_threshhold = _z,
                            zscore_buy_threshhold = -_z, trade_size=_ts,
                            position_limit=_pl, max_loss=_ml*cap,
                                starting_capital=cap)
                        _sr = self.sortino_ratio(_trial['returns_%'], annual_risk_free_rate=rf)
                        params = {}
                        params['zscore buy'] = - _z
                        params['zscore sell'] = _z
                        params['trade size'] = _ts
                        params['position limit'] = _pl
                        params['max loss'] = _ml*cap 
                        _results[_sr] = params

        bst_sr = max(_results)
        bst_params = _results[bst_sr]
        #print(datetime.datetime.now())
        return bst_params