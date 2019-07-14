# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 20:17:02 2019

@author: HP
"""

# *_*coding:utf-8 *_*  
from atrader import *  
from keras.models import *
import pandas as pd
import numpy as np


######### preprocessing functions  ################

def get_asset_code(df,name_col,thresh_hold):
    '''
    get assets that have trade records more than thresh_hold
    '''
    count_trade = df.groupby('name',as_index=False).agg({'close':'count'})
    count_trade = count_trade[count_trade.close>thresh_hold]
    count_trade['is_ST_or_S'] = count_trade['name'].apply(lambda x: 1 \
                                                          if x.startswith('*') or x.startswith('S') else 0)
    count_trade = count_trade[count_trade.is_ST_or_S==0]
    return count_trade

def get_valid_asset(df,asset_code,last_date):
    '''
    get assets whose last trade time are later than last_date
    '''
    df = df[df.name.isin(asset_code)]
    df = df.sort_values(['time'])
    last_trade_time = df.groupby(['name'],as_index=False)['time'].last()
    last_trade_time['time'] = pd.to_datetime(last_trade_time['time'], format='%Y-%m-%d')
    last_trade_time = last_trade_time[last_trade_time.time>=pd.to_datetime(last_date)]
    return last_trade_time


def read_data():
    code = get_code_list_set('SZAG','2005-01-01','2019-05-31')
    data = pd.read_csv('../data/szag_050101_190531.csv')
    data = data.merge(code[['code','name']], on='code')
    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values(['name','time'])
    asset_code = get_asset_code(data,'name',3000)
    valid_asset = get_valid_asset(data, asset_code.name.unique(), '2019-05-28 15:00:00')
    data = data.merge(valid_asset[['name']], on='name')
    code = data['code'].unique()
    return data,code


# define backtest time range
begin_date  = '2019-03-01'
end_date='2019-05-30'

def init(context: Context):  
    # 设置初始资金为100万  
    set_backtest(initial_cash=1000000)  
    
    # 注册1day的数据  
    reg_kdata(frequency='day', fre_num=1)  
    
    context.trade_flag = 0 
    
def on_data(context: Context):  
    # context.reg_kdata根据注册的先后顺序标记，0代表第一个注册的频率，为1day  
    
    # 获取 1day 注册的数据，并获取3个最新的1day数据,并以dataframe的形式输出  
    df = get_reg_kdata(reg_idx=context.reg_kdata[0], length=1, df=True, fill_up=True)    
    
    # trade every 10 days
    if context.trade_flag % 10 == 0:
    
        # load model
        model = load_model('model/test_baseline.h5')
        
        # predict confidence
        feats = ['open','high','low','close']
        df['confidence'] = model.predict(df[feats])
        confidence = df[['target_idx','confidence']].sort_values('confidence',ascending=False)
        confidence = confidence[:1]
        target_idx = confidence['target_idx'].values
        
        # 平前10天所有仓
        order_close_all(account_idx=0)
        
        # 买入开仓，市价委托
        order_percent(account_idx=0, target_idx=target_idx[0],percent=1,side=1,position_effect=1,order_type=2)
        
        context.trade_flag = 1
        print(df.time.max())
    
    else:
        context.trade_flag += 1

        
if __name__ == '__main__':  
    data,code = read_data()
    run_backtest(strategy_name='example_test', file_path='NN_baseline.py', target_list=code[:20], frequency='day',begin_date=begin_date, end_date='2019-05-30')