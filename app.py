import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import utils.frequencies as frequencies
import utils.mixed_modes_utils as mixed_modes_utils
import os, requests

from plotly.colors import sample_colorscale
from astropy.timeseries import LombScargle
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
from os import path

def Lor_model(pds, peak):
    return peak.height / (1 + ((pds.frequency.values - peak.frequency)/peak.linewidth)**2)

def sinc2_model(pds, peak):
    deltanu = np.mean(np.diff(pds.frequency.values))
    return peak.height * np.sinc((pds.frequency.values - peak.frequency)/deltanu)**2

def fit_model(pds, peaks):

    model = np.ones_like(pds.frequency.values)

    for i in range(len(peaks)):
        if np.isfinite(peaks.linewidth.iloc[i]):
            model += Lor_model(pds, peaks.iloc[i,])
        else:
            model += sinc2_model(pds, peaks.iloc[i, ])
    return model


def prepare_l1_peaks(peaks: pd.DataFrame, summary: pd.DataFrame,
                     AIC_cut: [float] = 0.0, height_cut: [float] = 0.0) -> pd.DataFrame:
    peaks['x'] = ((peaks['frequency'] % summary['DeltaNu'].values - summary['eps_p'].values) / summary['DeltaNu'].values) % 1
    # Don't want to include any modes near l=0 or 2s, this is why this and the step in the next cell is performed.
    x_range = [(np.minimum(np.min(peaks.loc[peaks['l'] == 0, 'x']), np.min(peaks.loc[peaks['l'] == 2, 'x'])) - 0.05) % 1,
               (np.maximum(np.max(peaks.loc[peaks['l'] == 0, 'x']), np.max(peaks.loc[peaks['l'] == 2, 'x'])) + 0.05) % 1]
    
    l1_peaks = peaks.loc[(peaks.l == 1) | ~np.isfinite(peaks.l), ]
    l1_peaks['x'] = ((l1_peaks['frequency'] % summary['DeltaNu'].values - summary['eps_p'].values) / summary['DeltaNu'].values) % 1
    if x_range[0] < x_range[1]:
        l1_peaks = l1_peaks.loc[(l1_peaks['x'] < x_range[1]) | (l1_peaks['x'] > x_range[0]), ] # changed to OR for HeB
    else:
        print(x_range)
        l1_peaks = l1_peaks.loc[(l1_peaks['x'] > x_range[1]) & (l1_peaks['x'] < x_range[0]), ]


    l1_peaks = l1_peaks.loc[(l1_peaks['height'] > height_cut), ]
    l1_peaks = l1_peaks.loc[(l1_peaks['AIC'] > AIC_cut), ]

    return l1_peaks



def create_stretched_echelle(dpi1, q, eps_g, drot, rot_components, inp_freqs, pds_freq, numax, dnu,
                            eps_p, alpha):
    freqs_dummy = frequencies.Frequencies(frequency=pds_freq,
                                    numax=numax, 
                                    delta_nu=dnu if np.isfinite(dnu) else None, 
                                    epsilon_p=eps_p if np.isfinite(eps_p) else None,
                                    alpha=alpha if np.isfinite(alpha) else None)

    try:
        splitting=drot
    except:
        splitting = 0
    params = {'calc_l0': True, 
                'calc_l2': True, 
                'calc_l3': False, 
                'calc_nom_l1': True, 
                'calc_mixed': True, 
                'calc_rot': False, 
                'DPi1': dpi1,
                'coupling': q,
                'eps_g': eps_g,
                'l': 1, 
                }

    cand_dpi = dpi1
    freqs_dummy(params)
    freqs_dummy.generate_tau_values()

    real_tau = mixed_modes_utils.peaks_stretched_period(inp_freqs, 
                                                             freqs_dummy.frequency, 
                                                             freqs_dummy.tau)
    real_tau = real_tau - freqs_dummy.DPi1*(freqs_dummy.shift)


    freqs_p1 = freqs_dummy.l1_mixed_freqs + freqs_dummy.l1_zeta * splitting
    freqs_n1 = freqs_dummy.l1_mixed_freqs - freqs_dummy.l1_zeta * splitting

    tau_p1 = mixed_modes_utils.peaks_stretched_period(freqs_p1, freqs_dummy.frequency, freqs_dummy.tau)
    tau_n1 = mixed_modes_utils.peaks_stretched_period(freqs_n1, freqs_dummy.frequency, freqs_dummy.tau)

    model_freqs = np.c_[freqs_dummy.l1_mixed_freqs, freqs_p1, freqs_n1]
    model_tau = np.c_[freqs_dummy.l1_mixed_tau, tau_p1, tau_n1]

    plot_tau = np.mod(real_tau, freqs_dummy.DPi1)
    plot_tau[plot_tau > freqs_dummy.DPi1/2] -= freqs_dummy.DPi1

    if rot_components == 'Singlet':
        X = np.c_[model_tau[:,0], model_freqs[:,0]]
        rot_id = np.zeros(len(model_tau))
    elif rot_components == 'Doublet':
        X = np.c_[np.r_[model_tau[:,1] - freqs_dummy.shift * freqs_dummy.DPi1, 
                        model_tau[:,2] - freqs_dummy.shift * freqs_dummy.DPi1], 
                  np.r_[model_freqs[:,1], 
                        model_freqs[:,2]]]
        rot_id = np.concatenate([np.ones(len(model_tau)), -np.ones(len(model_tau))], axis=0)
    else: # triplet
        X = np.c_[np.r_[model_tau[:,0],
                        model_tau[:,1] - freqs_dummy.shift * freqs_dummy.DPi1, 
                        model_tau[:,2] - freqs_dummy.shift * freqs_dummy.DPi1], 
                  np.r_[model_freqs[:,0],
                        model_freqs[:,1], 
                        model_freqs[:,2]]]
        rot_id = np.concatenate([np.zeros(len(model_tau)), np.ones(len(model_tau)), -np.ones(len(model_tau))],
                               axis=0)

    plot_model_tau = np.mod(X[:,0], freqs_dummy.DPi1)
    plot_model_tau[plot_model_tau > freqs_dummy.DPi1/2] -=  freqs_dummy.DPi1
    
    ## calculating the cost between frequencies and tau directly ##

    c1 = np.vstack((X[:,0]/freqs_dummy.DPi1, (X[:,1]-freqs_dummy.numax)/freqs_dummy.delta_nu)).T
    c2 = np.vstack((real_tau/freqs_dummy.DPi1, (inp_freqs-freqs_dummy.numax)/freqs_dummy.delta_nu)).T
    kost_matrix = cdist(c1, c2, 'euclidean') 
    row_ind, col_ind = linear_sum_assignment(kost_matrix) 
    
    return plot_model_tau, plot_tau, X[:,1], inp_freqs, row_ind, rot_id, freqs_dummy



def create_psps(dpi, q, pds_freq, pds_power, freqs):
    
    up_bound, low_bound = 200, 20 # RGB
    # up_bound, low_bound = 400, 100 # HeB
    

    params = {'calc_l0': True, 
                'calc_l2': True,
                'calc_l3': False, 
                'calc_nom_l1': True, 
                'calc_mixed': False,
                'calc_rot': False, 
                'DPi1': dpi,
                'coupling': q,
                'eps_g': 0.0, 
                'l': 1,
                }
    freqs(params)

    if pds_freq.min() < freqs.l0_freqs.min():
        zeta = freqs.zeta[pds_freq >= freqs.l0_freqs.min()]
        power = pds_power[pds_freq >= freqs.l0_freqs.min()].values
        freq = pds_freq[pds_freq >= freqs.l0_freqs.min()].values
    else:
        power = pds_power
        freq = pds_freq
        zeta = freqs.zeta

    new_frequency, tau, zeta = mixed_modes_utils.stretched_pds(freq, 
                                                               zeta)

    fr = np.arange(1/(up_bound), 1/(low_bound), 0.1/tau.max()) 
    ls = LombScargle(tau, power)
    PSD_LS = ls.power(fr)
    return 1/fr, PSD_LS
    
    
user = "mtyhon"
repo = "herokuapp"

url = "https://api.github.com/repos/{}/{}/git/trees/master?recursive=1".format(user, repo)
r = requests.get(url)
res = r.json()
## Initial Parse to get KICs in the folder ##
kic_list = []
for file in res["tree"]:
    if len(file["path"].split('/')) > 2:
        if file["path"].split('/')[2].isnumeric():
            kic_list.append(int(file["path"].split('/')[2]))

kic_list = np.unique(kic_list)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title ='Rotation Viewer'
server = app.server

################# LAYOUT #####################

app.layout = html.Div([
    #### DROPDOWN AT TOP LEFT ####
    html.Div([ dcc.Dropdown(
                id='star',
                options=[{'label': 'KIC %d' %i, 'value': i} for i in kic_list],
    placeholder="KIC ID (select)", style={'padding-top': '2px', 'verticalAlign': 'middle'},
    value=kic_list[0]), 
                 ],
        style={'width': '50%', 'display': 'inline-block','height': '40px',
               'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        #'padding-top': '5px'
              }),


    #### DROPDOWN AT TOP RIGHT ####
    html.Div([ dcc.Dropdown(
                id='rot_components',
                options=[{'label': i, 'value': i} for i in ['Singlet', 'Doublet', 'Triplet']],
    placeholder="Number of rotational components (select)", style={'padding-top': '2px', 'verticalAlign': 'middle'}), 
                 ],
        style={'width': '50%', 'display': 'inline-block','height': '40px',
               'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        #'padding-top': '5px'
              }),
    
    
    #### GRID ####
    html.Div(
        className="row",
        children=[
            
            ### LEFT COLUMN ###
            html.Div(
                className="six columns",
                children=html.Div([
                    
                    ### ROW ONE ###
                    dcc.Graph(
                            id='dpi1-eps_g',
                        ),
                    ### ROW TWO ###
                    dcc.Graph(
                            id='dpi1-q',
                        ),
                    ### ROW THREE ###
                    dcc.Graph(
                            id='dpi1-drot',
                        ),           
                    ])
            ),
            
            ### RIGHT COLUMN ###
            html.Div(
                className="six columns",
                children=html.Div([
                    
                    ### ROW ONE ###
                    dcc.Graph(
                        id='stretched_period_echelle',
                    ),
                    
                    ### ROW TWO ###
                    dcc.Graph(
                        id='psxps'
                    ),

                ])
            )         
        ]
    )
    ,
    ### TOGGLE FREQUENCY ##
    html.Div(
        dcc.Checklist(
        id='freq_toggle',
        options=[
            {'label': 'Toggle Peakbagged Modes', 'value': 'real'},
            {'label': 'Toggle Model Frequencies', 'value': 'model'},
        ],
        value=['real', 'model'],
        labelStyle={'display': 'inline-block',"margin-right": "25px"}
    ),        style={'width': '100%', 'display': 'inline-block',
                'verticalAlign': 'top', 'horizontalAlign': 'middle',
                'padding-bottom': '0px', 'padding-top': '10px',
              })
    
    ## put new DIV here
    ,html.Div(dcc.Graph(id='PowerSpectrum'))
])


################# UPDATE FUNCTIONS ###################



def format_scatter(df, var2, starid, best_index, pds_l023_removed, prepped_l1_peaks, summary):    
    data_ = [dict(
        x=df.DPi1.values,
        y=df[var2].values,
        mode='markers',
        marker={'color': df.Loss.values, 'size': 4},
        hovertemplate =
        '<b>DPi1</b>: %{x:.4f}s'+
        '<br><b>%s</b>:'%var2+' %{y:.3f}<br><extra></extra>',
    )]
    
    yaxis_title = {'q': 'Coupling Factor q',
                  'eps_g': 'g-mode Phase Offset',
                  'drot': 'Core Rotation Rate (uHz)'}
    series_range = {'q': [min(df.q), max(df.q)],
                  'eps_g': [min(df.eps_g), max(df.eps_g)],
                  'drot': [min(df.drot), max(df.drot)]}
    

    return {'data': data_,
        'layout': {'coloraxis': 
         {'colorbar': {'title': {'text': 'Loss'}}, 
          'colorscale': [[0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1, '#f0f921']], 'showscale': False}, 
         'legend': {'tracegroupgap': 0}, 
          "height": 210,
         'margin': {'b': 20, 'l': 10, 'r': 0, 't': 10},
         'plot_bgcolor': 'ivory', 
         'template': {'layout': {'annotationdefaults': {'arrowcolor': '#2a3f5f', 'arrowhead': 0, 'arrowwidth': 1},
                                 'autotypenumbers': 'strict', 
                                 'coloraxis': {'colorbar': {'outlinewidth': 0, 'ticks': ''}},
                                 'colorscale': {'diverging': [[0, '#8e0152'], [0.1, '#c51b7d'], [0.2, '#de77ae'],
                                                              [0.3, '#f1b6da'], [0.4, '#fde0ef'], 
                                                              [0.5, '#f7f7f7'], [0.6, '#e6f5d0'],
                                                              [0.7, '#b8e186'], [0.8, '#7fbc41'], 
                                                              [0.9, '#4d9221'], [1, '#276419']], 
                                                'sequential': [[0, '#0d0887'], [0.1111111111111111, '#46039f'], 
                                                               [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'],
                                                               [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1, '#f0f921']], 'sequentialminus': [[0, '#0d0887'], [0.1111111111111111, '#46039f'],
                                                               [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'],
                                                               [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'],
                                                               [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'],
                                                               [0.8888888888888888, '#fdca26'], [1, '#f0f921']]},
                                 'colorway': ['#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A', '#19d3f3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'], 'font': {'color': '#2a3f5f'}, 'geo': {'bgcolor': 'white', 'lakecolor': 'white', 'landcolor': '#E5ECF6', 'showlakes': True, 'showland': True, 'subunitcolor': 'white'}, 
                                 'hoverlabel': {'align': 'left'}, 'hovermode': 'closest', 'mapbox': {'style': 'light'}, 'paper_bgcolor': 'white', 'plot_bgcolor': '#E5ECF6', 'polar': {'angularaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}, 
                                 'bgcolor': '#E5ECF6', 'radialaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}}, 'scene': {'xaxis': {'backgroundcolor': '#E5ECF6', 'gridcolor': 'white', 'gridwidth': 2, 'linecolor': 'white', 'showbackground': True, 'ticks': '', 'zerolinecolor': 'white'}, 'yaxis': {'backgroundcolor': '#E5ECF6', 'gridcolor': 'white', 'gridwidth': 2, 'linecolor': 'white', 'showbackground': True, 'ticks': '', 'zerolinecolor': 'white'}, 'zaxis': {'backgroundcolor': '#E5ECF6', 'gridcolor': 'white', 'gridwidth': 2, 'linecolor': 'white', 'showbackground': True, 'ticks': '', 'zerolinecolor': 'white'}}, 'shapedefaults': {'line': {'color': '#2a3f5f'}}, 'ternary': {'aaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}, 'baxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}, 'bgcolor': '#E5ECF6', 'caxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}}, 
                                 'title': {'x': 0.05}, 'xaxis': {'automargin': True, 'gridcolor': 'white', 'linecolor': 'white', 'ticks': '', 'title': {'standoff': 15}, 'zerolinecolor': 'white', 'zerolinewidth': 2}, 
                                 'yaxis': {'automargin': True, 'gridcolor': 'white', 'linecolor': 'white', 'ticks': '', 'title': {'standoff': 15}, 'zerolinecolor': 'white', 'zerolinewidth': 2}}}, 'title': {'font': {'color': 'white'}, 'x': 0.2, 'y': 0.99}, 
                                 'xaxis': {'anchor': 'y', 'domain': [0, 1], 'gridcolor': 'gainsboro', 'range': [min(df.DPi1), max(df.DPi1)], 'title': {'standoff': 10, 'text': 'Period Spacing (s)'}, 'type': 'linear'}, 'yaxis': {'anchor': 'x', 'domain': [0, 1], 
                                 'gridcolor': 'gainsboro', 'range': series_range[var2], 'title': {'standoff': 10, 'text': yaxis_title[var2]}, 'type': 'linear'
                                 },'autosize': True,
              'shapes': [{'line': {'color': None, 'dash': 'longdashdot', 'width': 0, 'opacity': 0},
                         'type': 'line',
                         'x0': min(df.DPi1),
                         'x1': min(df.DPi1),
                         'xref': 'x',
                         'y0': series_range[var2][0],
                         'y1': series_range[var2][0], # Save metadata hack!
                         'yref': 'y',
                        'starid': starid,
                         'init_var_value': df[var2].values[best_index],
                         'init_dpi_value': df.DPi1.values[best_index],
                         'pds_l023_removed_freq': pds_l023_removed.frequency.values,
                          'pds_l023_removed_power': pds_l023_removed.power.values,
                          'inp_freqs' : prepped_l1_peaks.frequency.values,
                         'numax': summary.numax.values,
                         'dnu': summary.DeltaNu.values,
                         'eps_p': summary.eps_p.values,
                         'alpha': summary.alpha.values}]
                  },
           }


def format_stretched(model_tau, real_tau, model_freqs, real_freqs, dpi1_value, q_value, eps_g_value, drot_value,
                    save_dpi_1, save_dpi_2, save_dpi_3,rot_idx, star_sample):
    
    data_ = [dict(
        x=model_tau[rot_idx == -1]/dpi1_value,
        y=model_freqs[rot_idx == -1],
        name='Model l=-1',
        mode='markers',
        marker={'color': 'red',
             'size': 4,
               'opacity': 0.4},
        hovertemplate =
        '<b>Model tau mod DPi1</b>: %{x:.1f}s'+
        '<br><b>Model Freq (uHz)</b>: %{y:.4f}<extra></extra>',
    ),
        dict(
        x=model_tau[rot_idx == 0]/dpi1_value,
        y=model_freqs[rot_idx == 0],
        name='Model l= 0',
        mode='markers',
        marker={'color': 'cyan',
             'size': 4,
               'opacity': 0.4},
        hovertemplate =
        '<b>Model tau mod DPi1</b>: %{x:.1f}s'+
        '<br><b>Model Freq (uHz)</b>: %{y:.4f}<extra></extra>',
    ),
             
        dict(
        x=model_tau[rot_idx == 1]/dpi1_value,
        y=model_freqs[rot_idx == 1],
        name='Model l=+1',
        mode='markers',
        marker={'color': 'green',
             'size': 4,
               'opacity': 0.4},
        hovertemplate =
        '<b>Model tau mod DPi1</b>: %{x:.1f}s'+
        '<br><b>Model Freq (uHz)</b>: %{y:.4f}<extra></extra>',
    ),
               
            dict(
        x=real_tau/dpi1_value,
        y=real_freqs,
        name='Observed',
        mode='markers',
        marker={'color': 'royalblue',
             'size': 6},
        hovertemplate =
        '<b>Real tau mod DPi1</b>: %{x:.1f}s'+
        '<br><b>Real Freq (uHz)</b>: %{y:.4f}<extra></extra>', 
    )]
   
    return {
        'data': data_,
        
        'layout': {
            'annotations': [{'x': 0.007, 'y': 0.65, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(250, 250, 250, 0.5)',
                'font': {'size': 12},
                'text': 'Stretched<br>Period Echelle<br><br>dpi1: %.4fs<br>q: %.3f <br>eps_g: %.3f <br>d_rot: %.3f uHz' %(dpi1_value, q_value, eps_g_value, drot_value)
            }],

            'yaxis': {'type': 'linear',
                     "title": {'text': "Frequency (uHz)", 'standoff': 2},
                     "range": [min(real_freqs) - 1, max(real_freqs) + 1]},
            'xaxis': {'showgrid': False, 
                     "title": {'text': "tau modulo DPi1 (s)", 'standoff': 10},
                     "range": [-0.51, 0.51]},
            'hoverlabel': {'bgcolor': 'gray',
                           'namelength': 1,
                          'font': {'color': 'white'}},
            'legend': {'x': 0.85, 'y': 0.025, 'bgcolor': 'rgba(250, 250, 250, 0.75)',
                      'bordercolor': 'black', 'borderwidth': 1},
            'margin': {'l': 40, 'b': 40, 't': 10, 'r': 0},
            "height": 430,
            "plot_bgcolor": 'ivory',
            "uirevision": True,  # when we update don't move out of zoom     
            'shapes': [{'line': {'color': None, 'dash': 'longdashdot', 'width': 0, 'opacity': 0},
             'type': 'line',
             'x0': 0,
             'x1': 0,
             'xref': 'x',
             'y0': 0,
             'y1': 0, # Save metadata hack! You can even define nonsense things in the dict to preserve stuff
             'yref': 'y',
                'save_dpi_1': save_dpi_1,
                'save_dpi_2': save_dpi_2,
                'save_dpi_3': save_dpi_3,
                'star_id': star_sample}],
        }

    }


def format_psps(period, PSD_LS, dpi, tracecolor, xrange):
    
    xrange[0], xrange[1] = xrange[0] - 1.2*(xrange[1] - xrange[0]), xrange[1] + 1.2*(xrange[1] - xrange[0])
    data_ = [dict(
        x=period,
        y=PSD_LS,
        mode='line',
        line={'color': tracecolor,
             'width': 2,
             'coloraxis': 'coloraxis'},
        hovertemplate =
        '<b>DPi1</b>: %{x:.1f}s'+
        '<br><b>Power</b>: %{y:.4f}<extra></extra>', #<extra></extra> removes the 'Trace 0' in the hover
    )]
    psd_interval = PSD_LS[(period >= xrange[0]) & (period <= xrange[1])]
    return {
        'data': data_,
        
        'layout': {
            'annotations': [{'x': 0.85, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': ' PSxPS'}],

            'yaxis': {'type': 'linear',
                     "title": {'text': "Power", 'standoff': 15},
                     "range": [0, max(psd_interval)*1.15]},
            'xaxis': {'showgrid': False, 
                     "title": {'text': "Period Spacing (s)", 'standoff': 15},
                     "range": xrange},
            'hoverlabel': {'bgcolor': 'gray',
                           'namelength': 1,
                          'font': {'color': 'white'}},
            'margin': {'l': 50, 'b': 40, 't': 10, 'r': 0},
            "height": 200,
            'shapes': [{'line': {'color': 'black', 'dash': 'longdashdot', 'width': 1.5},
                 'type': 'line',
                 'x0': dpi,
                 'x1': dpi,
                 'xref': 'x',
                 'y0': 0,
                 'y1': max(PSD_LS)*1.15,
                 'yref': 'y'}],
                     
        }

    }


def format_spectrum(real_freqs, model_freqs, freq_toggle, rot_idx, pds_freq, pds_power): # model_freqs already matched
    data_ = [dict(
        x=pds_freq,
        y=pds_power,
        mode='line',
        line={'color': 'black',
             'width': 2},
        hovertemplate =
        '<b>Frequency (uHz)</b>: %{x:.2f}s'+
        '<br><b>Power</b>: %{y:.4f}<extra></extra>', #<extra></extra> removes the 'Trace 0' in the hover
    )]
    
    shapelist = []

    colordict = {0: 'cyan',
                -1: 'red',
                1: 'green'}
#     print([colordict[zz] for zz in rot_idx])    
    if 'real' in freq_toggle:   
        for l in real_freqs:
            linedict = {'line': {'color': 'royalblue', 'dash': 'dash', 'width': 1.25},
                         'type': 'line',
                        'layer': 'below',
                         'x0': l,
                         'x1': l,
                         'xref': 'x',
                         'y0': 0,
                         'y1': np.max(pds_power),
                         'yref': 'y'}
            shapelist.append(linedict)
    if 'model' in freq_toggle:  
        for k, l in enumerate(model_freqs):
            linedict = {'line': {'color': colordict[rot_idx[k]], 'dash': 'dot', 'width': 1.1},
                         'type': 'line',
                        'layer': 'below',
                         'x0': l,
                         'x1': l,
                         'xref': 'x',
                         'y0': 0,
                         'y1': np.max(pds_power),
                         'yref': 'y'}
            shapelist.append(linedict)        
            
    return {
        'data': data_,
        
        'layout': {
            'yaxis': {'type': 'linear',
                     "title": {'text': "Power (S/N)", 'standoff': 15},
                     "range": [0, np.max(pds_power)]},
            'xaxis': {'showgrid': False, 
                     "title": {'text': "Frequency (uHz)", 'standoff': 5},
                     "range": [np.min(pds_freq), np.max(pds_freq)]},
            'hoverlabel': {'bgcolor': 'gray',
                           'namelength': 1,
                          'font': {'color': 'white'}},
            'margin': {'l': 50, 'b': 50, 't': 10, 'r': 30},
            "height": 250,
            "uirevision": True,  # when we update don't move out of zoom        
            'shapes': shapelist,                     
        }

    }
    

@app.callback(
    [dash.dependencies.Output('dpi1-eps_g', 'figure'), dash.dependencies.Output('dpi1-q', 'figure'),
    dash.dependencies.Output('dpi1-drot', 'figure')],
    dash.dependencies.Input('star', 'value'))
def update_sample(samplename):
    ## Updates Star from whatever in the list, updates all figures
    kicz = samplename

    ## Subsequent Parse to Query Files for a Specific KICID ##
    user = "mtyhon"
    repo = "herokuapp"
    url = "https://api.github.com/repos/{}/{}/git/trees/master?recursive=1".format(user, repo)
    r = requests.get(url)

    for file in res["tree"]:
        if len(file["path"].split('/')) > 2:
            if file["path"].split('/')[2].isnumeric():
                if int(file["path"].split('/')[2]) == kicz:    
                    if file["path"].endswith('samples.csv'):
                        samples_filename = file["path"]
                        print('Samples Loaded!')
                    if file["path"].endswith('summary.csv'):
                        summary_filename = file["path"]
                    if file["path"].endswith('/pds_bgr.csv'):
                        pds_filename = file["path"]
                        print('PDS Loaded!')
                    if file["path"].endswith('peaksMLE.csv'):
                        print('Peaks Loaded!')
                        peaks_filename = file["path"]

    samples_url = "https://raw.githubusercontent.com/{}/{}/master/{}".format(user, repo, samples_filename)
    summary_url = "https://raw.githubusercontent.com/{}/{}/master/{}".format(user, repo, summary_filename)
    pds_url = "https://raw.githubusercontent.com/{}/{}/master/{}".format(user, repo, pds_filename)
    peaks_url = "https://raw.githubusercontent.com/{}/{}/master/{}".format(user, repo, peaks_filename)

    summary = pd.read_csv(summary_url)
    pds = pd.read_csv(pds_url)
    peaks = pd.read_csv(peaks_url)
    samples = pd.read_csv(samples_url) 
    inp_df = samples
    
    prepped_l1_peaks = prepare_l1_peaks(peaks, summary=summary, AIC_cut=5)
    pds = pds.loc[abs(pds['frequency'].values - summary['numax'].values) < 3 * summary['sigmaEnv'].values, ]
    peaks = peaks.loc[abs(peaks.frequency.values - summary.numax.values) < 3*summary.sigmaEnv.values, ]
    l023_peaks = peaks.loc[(peaks.l == 0) | (peaks.l == 2) | (peaks.l == 3)]
    l0_peaks = peaks.loc[(peaks.l==0), ]
    l1_peaks = peaks.loc[(peaks.l == 1)  | (np.isfinite(peaks.l) == False)]
    l2_peaks = peaks.loc[(peaks.l==2), ]
    pds_l023_removed = pds.assign(power = pds.power / fit_model(pds, l023_peaks))
    inp_freqs = prepped_l1_peaks['frequency'].values
    prepped_l1_peaks['weights'] = (prepped_l1_peaks['amplitude']/np.sum(prepped_l1_peaks['amplitude']))*1000.
    file_best_index = int(samples_filename.split('-')[1].split('.')[0])
    
    return format_scatter(inp_df, var2='eps_g', starid=kicz, best_index=file_best_index, pds_l023_removed = pds_l023_removed, prepped_l1_peaks=prepped_l1_peaks, summary=summary), \
format_scatter(inp_df, var2='q', starid=kicz, best_index=file_best_index, pds_l023_removed = pds_l023_removed, prepped_l1_peaks=prepped_l1_peaks, summary=summary), \
format_scatter(inp_df, var2='drot', starid=kicz, best_index=file_best_index, pds_l023_removed = pds_l023_removed, prepped_l1_peaks=prepped_l1_peaks, summary=summary)


@app.callback([dash.dependencies.Output('stretched_period_echelle', 'figure'),
               dash.dependencies.Output('psxps', 'figure'), dash.dependencies.Output('PowerSpectrum', 'figure')],
    [dash.dependencies.Input('dpi1-eps_g', 'clickData'), dash.dependencies.Input('dpi1-q', 'clickData'),
    dash.dependencies.Input('dpi1-drot', 'clickData'), dash.dependencies.Input('dpi1-eps_g', 'figure'),
     dash.dependencies.Input('dpi1-q', 'figure'), dash.dependencies.Input('dpi1-drot', 'figure'),
    dash.dependencies.Input('stretched_period_echelle', 'figure'), dash.dependencies.Input('rot_components', 'value'),
    dash.dependencies.Input('freq_toggle', 'value')])
def update_stretched(clickData_dpi_epsg, clickData_dpi_q, clickData_dpi_drot,
                    figure_dpi_epsg, figure_dpi_q, figure_dpi_drot, stretched_figure, 
                     rot_components, freq_toggle):
    
    colorlist = figure_dpi_epsg['data'][0]['marker']['color'] # loss values
    star_sample = figure_dpi_epsg['layout']['shapes'][0]['starid']   
    pds_freq = np.array(figure_dpi_epsg['layout']['shapes'][0]['pds_l023_removed_freq'])
    pds_power = np.array(figure_dpi_epsg['layout']['shapes'][0]['pds_l023_removed_power'])
    inp_freqs = np.array(figure_dpi_epsg['layout']['shapes'][0]['inp_freqs'])
    numax = figure_dpi_epsg['layout']['shapes'][0]['numax'][0]
    dnu = figure_dpi_epsg['layout']['shapes'][0]['dnu'][0]
    eps_p = figure_dpi_epsg['layout']['shapes'][0]['eps_p'][0]
    alpha = figure_dpi_epsg['layout']['shapes'][0]['alpha'][0]
    
#     print(pds_freq, pds_power)
    
    if rot_components is None:
        rot_components = 'Triplet' # The default
        
    if stretched_figure is not None: # An existing figure already exists
        old_dpi1 = stretched_figure['layout']['shapes'][0]['save_dpi_1']
        old_dpi2 = stretched_figure['layout']['shapes'][0]['save_dpi_2']
        old_dpi3 = stretched_figure['layout']['shapes'][0]['save_dpi_3']
        old_star_sample = stretched_figure['layout']['shapes'][0]['star_id']
    else:
        old_dpi1 = old_dpi2 = old_dpi3 = old_star_sample = 0
    
    
    if (clickData_dpi_epsg is not None) and (old_star_sample == star_sample):
        dpi1_value1 = clickData_dpi_epsg['points'][0]['x']
        eps_g_value = clickData_dpi_epsg['points'][0]['y']
        markercol =  clickData_dpi_epsg['points'][0]['marker.color']
        _c = (markercol - np.min(colorlist)) / (np.max(colorlist) - np.min(colorlist))
        try:
            tracecolor1 = sample_colorscale(figure_dpi_epsg['layout']['coloraxis']['colorscale'],
                                               [_c], low=0.0, high=1.0, colortype='rgb')[0] # throws error on sample change
        except:
            tracecolor1 = stretched_figure['data'][0]['marker']['color'] # access previously used color
    else:
        dpi1_value1 = figure_dpi_epsg['layout']['shapes'][0]['init_dpi_value']
        eps_g_value = figure_dpi_epsg['layout']['shapes'][0]['init_var_value']
        tracecolor1 = 'red'
        
    if dpi1_value1 != old_dpi1:
        dpi1_value = dpi1_value1
        tracecolor = tracecolor1
    
   
    if (clickData_dpi_q is not None) and (star_sample == figure_dpi_q['layout']['shapes'][0]['starid']):
        dpi1_value2 = clickData_dpi_q['points'][0]['x']
        q_value = clickData_dpi_q['points'][0]['y']
        markercol =  clickData_dpi_q['points'][0]['marker.color']
        _c = (markercol - np.min(colorlist)) / (np.max(colorlist) - np.min(colorlist))
        
        try:
            tracecolor2 = sample_colorscale(figure_dpi_epsg['layout']['coloraxis']['colorscale'],
                                       [_c], low=0.0, high=1.0, colortype='rgb')[0]  
        except:
            tracecolor2 = stretched_figure['data'][0]['marker']['color'] # access previously used color  
            
    else:
        dpi1_value2 = figure_dpi_q['layout']['shapes'][0]['init_dpi_value']
        q_value = figure_dpi_q['layout']['shapes'][0]['init_var_value']
        tracecolor2 = 'red'
        
    if dpi1_value2 != old_dpi2:
        dpi1_value = dpi1_value2
        tracecolor = tracecolor2
           
        
    if (clickData_dpi_drot is not None) and (star_sample == figure_dpi_drot['layout']['shapes'][0]['starid']):
        dpi1_value3 = clickData_dpi_drot['points'][0]['x']
        drot_value = clickData_dpi_drot['points'][0]['y']
        markercol =  clickData_dpi_drot['points'][0]['marker.color']
        _c = (markercol - np.min(colorlist)) / (np.max(colorlist) - np.min(colorlist))
        try:
            tracecolor3 = sample_colorscale(figure_dpi_epsg['layout']['coloraxis']['colorscale'],
                                       [_c], low=0.0, high=1.0, colortype='rgb')[0]  
        except:
            tracecolor3 = stretched_figure['data'][0]['marker']['color'] # access previously used color  
    else:
        dpi1_value3 = figure_dpi_drot['layout']['shapes'][0]['init_dpi_value']
        drot_value = figure_dpi_drot['layout']['shapes'][0]['init_var_value']
        tracecolor3 = 'red'        
       
    if dpi1_value3 != old_dpi3:
        dpi1_value = dpi1_value3
        tracecolor = tracecolor3  
        
    if (dpi1_value1 == old_dpi1) & (dpi1_value2 == old_dpi2) & (dpi1_value3 == old_dpi3):
        dpi1_value = dpi1_value1
        tracecolor = stretched_figure['data'][0]['marker']['color'] # access previously used color  

        
    model_tau, real_tau, model_freqs, real_freqs, match_ind, rot_idx, freqs_class = create_stretched_echelle(dpi1_value, 
                                                                                                q_value,
                                                                                                eps_g_value, 
                                                                                                drot_value, 
                                                                                                rot_components,
                                                                                               inp_freqs,
                                                                                                pds_freq, 
                                                                                                numax, dnu,
                                                                                                eps_p, alpha) 
    period, PSD_LS = create_psps(dpi1_value, q_value, pds_freq, pds_power, freqs_class)

    return format_stretched(model_tau, real_tau, model_freqs, real_freqs,
                            dpi1_value, q_value, eps_g_value,
                            drot_value, dpi1_value1, dpi1_value2, dpi1_value3, rot_idx, star_sample),\
format_psps(period, PSD_LS, dpi1_value, tracecolor, figure_dpi_epsg['layout']['xaxis']['range']),\
format_spectrum(real_freqs, model_freqs[match_ind], freq_toggle, rot_idx[match_ind], pds_freq, pds_power)


if __name__ == '__main__':
    app.run_server(port=8055)
