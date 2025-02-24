# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:44:53 2025

@author: d_par
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:44:53 2025
@author: d_par
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import dash_bootstrap_components as dbc  # Import Bootstrap components
import sys
import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add Additional_Scripts to the Python path
sys.path.append(BASE_DIR)

from Additional_Scripts.App_Functions import (
    analyze_xg_vs_possession, analyze_xg_vs_formation,
    analyze_xg_vs_shots, analyze_xg_vs_month, analyze_xg_vs_time
)

# Load Data
file_path = os.path.join(os.path.dirname(__file__), "..", "Data", "team_data.csv")
team_data = pd.read_csv(file_path)

teams = sorted(team_data['Home'].unique())

# Initialize Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])  # Use CYBORG theme for modern look

# Layout
app.layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col(html.H1("Premier League xG Analysis", 
                        style={'textAlign': 'center', 'marginTop': '20px', 'color': '#17a2b8'}), 
                width=12)
    ]),

    # Dropdown Selection
    dbc.Row([
        dbc.Col(html.Label("Select a Team:", style={'fontSize': '20px', 'color': 'white'}), width=2),
        dbc.Col(dcc.Dropdown(
            id='team-selector',
            options=[{'label': team, 'value': team} for team in teams],
            value=teams[0],  
            clearable=False,
            style={'color': 'black'}
        ), width=4)
    ], justify='center', style={'marginBottom': '20px'}),

    # Grid Layout for Charts (2x3 Layout)
    dbc.Row([
        dbc.Col(dcc.Graph(id='xg-vs-possession-home', style={'height': '350px'}), width=6),
        dbc.Col(dcc.Graph(id='xg-vs-possession-away', style={'height': '350px'}), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='xg-vs-formation', style={'height': '350px'}), width=6),
        dbc.Col(dcc.Graph(id='xg-vs-shots', style={'height': '350px'}), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='xg-vs-month', style={'height': '350px'}), width=6),
        dbc.Col(dcc.Graph(id='xg-vs-time', style={'height': '350px'}), width=6),
    ]),
], fluid=True, style={'backgroundColor': '#121212', 'padding': '20px'})  # Dark Background

# Callbacks
@app.callback(
    Output('xg-vs-possession-home', 'figure'),
    Output('xg-vs-possession-away', 'figure'),
    Output('xg-vs-formation', 'figure'),
    Output('xg-vs-shots', 'figure'),
    Output('xg-vs-month', 'figure'),
    Output('xg-vs-time', 'figure'),
    Input('team-selector', 'value')
)
def update_charts(team):
    fig_home, fig_away = analyze_xg_vs_possession(team, team_data)
    fig2 = analyze_xg_vs_formation(team, team_data)
    fig3 = analyze_xg_vs_shots(team, team_data)
    fig4 = analyze_xg_vs_month(team, team_data)
    fig5 = analyze_xg_vs_time(team, team_data)
    return fig_home, fig_away, fig2, fig3, fig4, fig5

server = app.server  # Required for Render

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))

    

    
    