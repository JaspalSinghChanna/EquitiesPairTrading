import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
from analytics_module import Analytics
import datetime
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import zscore
from dash.exceptions import PreventUpdate

a = Analytics()
df2 = a.get_all_close_data()

app = dash.Dash(external_stylesheets=[dbc.themes.SOLAR])

app_title = html.Div([html.Div("Equity Pair Trading Tool", className="display-1"),])#html.H1("Pair Trading Tool")])


n_input = dbc.Input(id="n_pairs", placeholder="How many pairs do you want correlations for?",
                       type="number", value=10)

zscore_sell_in = dbc.Input(id="zscore_sell",
    placeholder="Zscore threshhold to signal sell.",type="number")

zscore_buy_in = dbc.Input(id="zscore_buy",
    placeholder="Zscore threshhold to signal buy.",type="number")

trade_size_in = dbc.Input(id="trade_size",
    placeholder="Volume of each trade.",type="number")

position_limit_in = dbc.Input(id="position_limit",
    placeholder="Max open position (volume) allowed.",type="number")

max_loss_in = dbc.Input(id='max_loss',
    placeholder='Max open loss before closing position until spread normalises.', type='number')

starting_capital_in = dbc.Input(id="starting_capital",
    placeholder="Starting capital.", type="number")

rf_in = dbc.Input(id='rf_in', value=5, type="number")

model_dropdown = dcc.Dropdown(options=['OLS', 'Kalman'],
                id='model_dropdown', placeholder='Select model.')

corr_table_heading = html.Div([html.H2("Most Positively Correlated Securities")])
corr_table = dash_table.DataTable(id='corr_table', page_size=5,
    row_selectable='single', selected_rows=[], #filter_action="native",
    #filter_options={"placeholder_text": "Filter column..."},
    #columns = ['Security 1', 'Security 2',' Correlation'],
    style_header={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'},
    style_data={'backgroundColor': 'rgb(50, 50, 50)','color': 'white'},)

pair_metrics_table = dash_table.DataTable(id='pair_metrics_table', page_size=5,
    style_header={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'},
    style_data={'backgroundColor': 'rgb(50, 50, 50)','color': 'white'},)

perf_metrics_table = dash_table.DataTable(id='perf_metrics_table', page_size=5,
    style_header={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'},
    style_data={'backgroundColor': 'rgb(50, 50, 50)','color': 'white'},)

date_range_row = dcc.DatePickerRange(
                id='date_range',
                min_date_allowed=datetime.date(2011, 1, 1),
                max_date_allowed=datetime.date(2024, 1, 1),
                clearable=True)

button = [dbc.Col(dbc.Button("Run backtest", color="primary", id='backtest_button'), className="d-grid gap-2 col-6 mx-auto"),
         dbc.Col(dbc.Button("Suggest parameters", color="light", id='suggest_button'), className="d-grid gap-2 col-6 mx-auto")]

loading_opts = dcc.Loading(
        id="loading-output",
        type="default",  # You can use "circle", "dot", "default", "cube", "circle-outside", or "circle-top"
        children=html.Div(id="opt_params")  # Your slow-loading component goes here
    )

@callback(
    [Output(component_id='corr_table', component_property='data'),],
    [Input(component_id='n_pairs', component_property='value'),
    Input(component_id='date_range', component_property='start_date'),
    Input(component_id='date_range', component_property='end_date'),]
)
def get_top_N_pairs(N, start_date, end_date):
    df = shorten_df(df2, start_date, end_date)
    pairs = a.get_pairs(df)
    return [pairs.head(N).to_dict('records')]

def shorten_df(df2, start_date, end_date):
    df = df2.copy()
    if start_date:
        st_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        df = df[st_dt:] 
    if end_date:
        ed_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        df = df[:ed_dt]
    return df
    
@app.callback(
    [Output('pair_selection_message', 'children'),],
    [Input('corr_table', 'derived_virtual_data'),
    Input('corr_table', 'derived_virtual_selected_rows')],
)
def update_selected_rows(rows, selection):
    if selection != [] and selection is not None:
        sec1 = rows[selection[0]]['Security 1']
        sec2 = rows[selection[0]]['Security 2']
        msg =  'You have selected: ' + sec1 + ' - ' + sec2
    else:
        msg = 'Select a pair using the table above.'
    return [html.H4(msg)]

@callback(
    [Output(component_id='pair_metrics_table', component_property='data'),
    Output('hedge_ratio_message','children'),
    Output('kalman_hr_graph','figure'),
    Output('OLS_zscore','figure'),
    Output('Kalman_zscore','figure')],
    [Input(component_id='date_range', component_property='start_date'),
    Input(component_id='date_range', component_property='end_date'),
    Input('corr_table', 'derived_virtual_data'),
    Input('corr_table', 'derived_virtual_selected_rows')]
)
def update_metrics_after_corr(start_date, end_date, rows, selection):
    if selection != [] and selection is not None:
        data = shorten_df(df2, start_date, end_date)
        sec1 = rows[selection[0]]['Security 1']
        sec2 = rows[selection[0]]['Security 2']
        global beta_ols, const_ols
        price_df, beta_ols, const_ols = a.fit_OLS(data[[sec1,sec2]], y=sec1, x=sec2)
        ols_mrs = a.get_mean_reversion_speed(price_df[['Residuals']])
        ols_adf_value = adfuller(price_df['Residuals'])[1]
        ols_hurst = a.hurst(price_df['Residuals'].values)
        global kal_df
        kal_df = a.run_Kalman_filter(price_df,y=sec1,x=sec2)
        kal_mrs = a.get_mean_reversion_speed(kal_df[['Residuals']])
        kal_adf_value = adfuller(kal_df['Residuals'])[1]
        kal_hurst = a.hurst(kal_df['Residuals'].values)
        _dict = {}
        _dict['Model'] = ['OLS','Kalman']
        _dict['Mean Reversion Speed'] = [ols_mrs, kal_mrs]
        _dict['ADF Test P-value'] = [ols_adf_value, kal_adf_value]
        _dict['Hurst Exponent'] = [ols_hurst, kal_hurst]
        _data = pd.DataFrame(_dict).round(5).to_dict("records")
        
        # now getting hedge ratios. easier in one function
        msg = f"OLS Hedge Ratio: {round(beta_ols,5)}"
        
        kal_fg = px.line(x=kal_df.index,y=kal_df['Beta'],
                         title="Evolution of Kalman Hedge Ratio", 
                         labels={'x': 'Time', 'y':'Beta'})
        
        # plotting residuals with z-score
        kal_res = px.line(x=kal_df.index,y=zscore(kal_df['Residuals']),
                         title="Kalman Model Spread", 
                         labels={'x': 'Time', 'y':'Z-Score for Spread'})
        
        ols_res = px.line(x=kal_df.index,y=zscore(price_df['Residuals']),
                         title="OLS Model Spread", 
                         labels={'x': 'Time', 'y':'Z-Score for Spread'})
        
        #for thresh in [1,1.5, -1, -1.5]:
        #    for fig in [kal_res,ols_res]:
        #        fig.add_hline(y=thresh,)
        
        return [_data, html.H5(msg), kal_fg, ols_res, kal_res]
    else:
        raise PreventUpdate

@callback(
    [Output(component_id='p_fig', component_property='figure'),
    Output(component_id='port_fig', component_property='figure'),
    Output(component_id='corr_fig', component_property='figure'),
    Output(component_id='perf_metrics_table', component_property='data'),],
    [Input('rf_in', 'value'),
    Input('backtest_button', 'n_clicks'),
    Input('corr_table', 'derived_virtual_data'),
    Input('corr_table', 'derived_virtual_selected_rows'),
    Input(component_id='model_dropdown', component_property='value'),
    Input(component_id='date_range', component_property='start_date'),
    Input(component_id='date_range', component_property='end_date'),
    Input(component_id='zscore_sell', component_property='value'),
    Input(component_id='zscore_buy', component_property='value'),
    Input(component_id='trade_size', component_property='value'),
    Input(component_id='position_limit', component_property='value'),
    Input(component_id='max_loss', component_property='value'),
    Input(component_id='starting_capital', component_property='value')]
)
def run_backtest(rf, n_clicks, rows, selection, model, start_date, end_date,
                 z_sell, z_buy, trade_sz, pos_lim, max_l, cap):
    global back_test_prev_clicks
    if n_clicks is not None:
        if n_clicks > back_test_prev_clicks:
            back_test_prev_clicks = n_clicks
            check = [selection, model, z_sell, z_buy, trade_sz, pos_lim, max_l, cap, rf]
            if len([x for x in check if x is None]) == 0:
                if selection != [] and selection is not None:
                    sec1 = rows[selection[0]]['Security 1']
                    sec2 = rows[selection[0]]['Security 2']
                if model=='OLS':
                    df = a.create_backtest_input_OLS(sec1, sec2, const_ols, beta_ols,
                                                     start_date, end_date)
                elif model=='Kalman':
                    df = a.create_backtest_input_Kalman(sec1, sec2, kal_df,
                                                        start_date, end_date)
                global exp_df1
                exp_df1 = df.copy()
                t2 = a.backtest(exp_df1, zscore_sell_threshhold = z_sell,
                    zscore_buy_threshhold = z_buy, trade_size=trade_sz,
                    position_limit=pos_lim, max_loss=max_l,
                        starting_capital=cap)
                buys = t2[t2['type']=='BUY']
                sells = t2[t2['type']=='SELL']
                p_fig = px.line(x=t2.index,y=t2['Open'],
                         title="Net Open Price/Spread", 
                         labels={'x': 'Time', 'y':'Price'})
                p_fig.add_scatter(x=sells.index, y=sells['price'], mode='markers', name='SELL',
                                  marker=dict(symbol='triangle-down', size=8, color='red'))
                p_fig.add_scatter(x=buys.index, y=buys['price'], mode='markers', name='BUY',
                                  marker=dict(symbol='triangle-up', size=8, color='green'))
                
                port_fig = px.line(x=t2.index,y=t2['position_plus_winnings'],
                         title="Cumulative Profit", 
                         labels={'x': 'Time', 'y':'Cumulative Profit (Including Open Position Value)'})
                #go.Scatter(x=df.index, y= df.z.where(df.z >= 1.5 ), mode = 'markers', marker =dict(symbol='triangle-down', size = 13, color = 'red'), 
                
                data = shorten_df(df2, start_date, end_date)
                rolling_corr = data[sec1].rolling(window=50).corr(data[sec2]).dropna()
                corr_fig = px.line(x=rolling_corr.index,y=rolling_corr,
                         title="Rolling Correlation (50 business days)", 
                         labels={'x': 'Time', 'y':'Correlation'})
                
                # evaluating performance
                sharpe = a.sharpe_ratio(t2['returns_%'], annual_risk_free_rate=rf)
                sortino = a.sortino_ratio(t2['returns_%'], annual_risk_free_rate=rf)
                max_drawdown = t2.drawdown.dropna().min()
                _dict = {}
                _dict['Sharpe Ratio'] = round(sharpe, 5)
                _dict['Sortino Ratio'] = round(sortino, 5)
                _dict['Max Drawdown (%)'] = round(max_drawdown*100, 5)
                
                return [p_fig, port_fig, corr_fig, [_dict]]
            else:
                raise PreventUpdate
        else:
            raise PreventUpdate
        #else:
        #    return ['no change...']
    else:
        back_test_prev_clicks = 0
        raise PreventUpdate
        #return ['waiting on clicks!']

        
@callback(
    [Output(component_id='opt_params', component_property='children'),],
    [Input('rf_in', 'value'),
    Input('suggest_button', 'n_clicks'),
    Input(component_id='max_loss', component_property='value'),
    Input(component_id='starting_capital', component_property='value')]
)
def suggest_params(rf, n_clicks, max_loss, cap):
    global suggest_prev_clicks
    if n_clicks is not None:
        if n_clicks > suggest_prev_clicks:
            suggest_prev_clicks = n_clicks
            check = [max_loss, cap, rf]
            if len([x for x in check if x is None]) == 0:
                #print('getting params')
                bst_params = a.get_optimised_params(exp_df1, cap, rf)
                #print(bst_params)
                return ['SUGGESTED PARAMS: ' + str(bst_params)]
        else:
            raise PreventUpdate
    else:
        suggest_prev_clicks = 0
        raise PreventUpdate
    

        
app.layout = dbc.Container([
    dbc.Row(app_title, className="text-center"),
    dbc.Row([html.Div(html.H5(
        "This equity pair trading tool analyses data across the S&P 500, Russell 2000 and Nasdaq 100 to "
        "look for pair relationships. We then use these relationships to backtest a mean-reversion trading"
        " strategy. First, we calculate correlations using end-of-day"
        " close prices and output the most positive correlations below."
        " Note that the pair you selected may change when you alter the date range."))],
        className='mt-3 text-left'),
    dbc.Row(html.Hr(), className="m-2"),
    dbc.Row(corr_table_heading, className="text-left"),
    dbc.Row(corr_table),
    dbc.Row([
        dbc.Col(html.Div([html.H4("Date Range (optional):")]), width=3),
        dbc.Col(date_range_row, width=4),# width={"order": 6})
        dbc.Col(html.Div([html.H4("Number of pairs in table:")]), width=3),
        dbc.Col(n_input, width=2),
        ]),
    dbc.Row(html.Hr(), className="mt-2"),
    dbc.Row([html.Div(id='pair_selection_message')]),
    dbc.Row([html.Div(html.H5(
        "For the selected pair above, we now run an OLS regression (y=Security 1, x=Security 2) and show the constant"
        " beta below. This provides an estimated optimal hedge ratio. However, the relationship may change over time."
        " The Kalman filter accounts for this, and a plot of the estimated changing beta values is shown below."))],
        className='mt-3 text-left'),
    dbc.Row([pair_metrics_table], className='m-2'),
    dbc.Row([html.Div(id='hedge_ratio_message')], className='mt-4'),
    dbc.Row([html.Div(html.H5("Kalman Filter Hedge Ratio:"))], className='mt-1'),
    dbc.Row([dcc.Graph(figure={}, id='kalman_hr_graph')], className="mb-3"),
    dbc.Row(html.Hr(), className="mt-4"),
    dbc.Row([html.Div(html.H2("Backtesting A Trading Strategy"))], className='mt-1'),
    dbc.Row([html.Div(html.H5(
        "Now we can determine a trading strategy based on signals from the OLS or the Kalman filter model."
        " We plot the spread for each model below. In our strategy, each unit of the basket we are trading "
        " contains 1 unit of Security 1 and beta units of Security 2, e.g. by purchasing one share of "
        "Security 1 and shorting beta shares of Security 2. The spread indicates both the signal "
        "and the price of the basket we are trading. The signal is still based on close prices, but trades are "
        "entered using the open price of the following day."))],
        className='mt-3 text-left'),
    dbc.Row([html.Div(html.H3("Distribution of Spread By Model"))], className='mt-1 text-center'),
    dbc.Row([
        dbc.Col(dcc.Graph(figure={}, id='OLS_zscore'), className="mb-3"),
        dbc.Col(dcc.Graph(figure={}, id='Kalman_zscore'), className="mb-3")
    ]),
    dbc.Row([html.Div(html.H3("Backtesting Parameters"))], className='mt-1 text-center'),
    dbc.Row([
        dbc.Col(dbc.Label('Model for backtesting:')),
        dbc.Col(model_dropdown, width=4),
        ]),
    dbc.Row([
        dbc.Col(dbc.Label('Z-score threshhold to signal buy (usually negative):')),
        dbc.Col(zscore_buy_in, width=4),
        ]),
    dbc.Row([
        dbc.Col(dbc.Label('Z-score threshhold to signal sell (usually positive):')),
        dbc.Col(zscore_sell_in, width=4)
        ]),
    dbc.Row([
        dbc.Col(dbc.Label('Volume of each trade (shares):')),
        dbc.Col(trade_size_in, width=4)
        ]),
    dbc.Row([
        dbc.Col(dbc.Label('Max open position allowed (shares):')),
        dbc.Col(position_limit_in, width=4)
        ]),
    dbc.Row([
        dbc.Col(dbc.Label('Max open loss before closing position until spread normalises ($):')),
        dbc.Col(max_loss_in, width=4)
        ]),
    dbc.Row([
        dbc.Col(dbc.Label('Starting capital ($):')),
        dbc.Col(starting_capital_in, width=4)
        ]),
    dbc.Row([
        dbc.Col(dbc.Label('Annualised risk-free interest rate (%):')),
        dbc.Col(rf_in, width=4),
        ]),
    dbc.Row([html.Div(html.H6(
        "All parameters must be defined before running backtest or getting suggested values."))],
        className='mt-3 text-center'),
    dbc.Row(button, className="m-3"),
    dbc.Row(loading_opts, className='my-2'),
    dbc.Row([html.Div(html.H3("Performance"))], className='mt-1 text-center'),
    dbc.Row(dcc.Graph(figure={}, id='p_fig'), className="mb-3"),
    #dbc.Row(loading_fig, className="mb-3"),
    dbc.Row(dcc.Graph(figure={}, id='port_fig'), className="mb-3"),
    dbc.Row(dcc.Graph(figure={}, id='corr_fig'), className="mb-3"),
    dbc.Row([html.Div(html.H3("Evaluation Metrics"))], className='mt-1 text-center'),
    dbc.Row(perf_metrics_table, className='mb-4'),
    dbc.Row(html.Hr(), className="m-4"),
            
])

if __name__ == "__main__":
    app.run_server(port=8051, debug=True)
