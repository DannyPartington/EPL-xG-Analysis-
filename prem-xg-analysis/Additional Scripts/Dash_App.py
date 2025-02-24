# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:44:53 2025

@author: d_par
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from App_Functions import (analyze_xg_vs_possession, analyze_xg_vs_formation,
                           analyze_xg_vs_shots, analyze_xg_vs_month, analyze_xg_vs_time)

team_data = pd.read_csv(r'C:\Users\d_par\OneDrive\Desktop\Danny\2025\Data Science\Portolio Projects\prem-xg-analysis\Notebooks')

teams = sorted(team_data['Home'].unique())


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Premier League xG Analysis", style={'textAlign': 'center'}),

    html.Label("Select a Team:"),
    dcc.Dropdown(
        id='team-selector',
        options=[{'label': team, 'value': team} for team in teams],
        value=teams[0],  
        clearable=False
    ),

    html.Br(),

    dcc.Graph(id='xg-vs-possession-home'),
    dcc.Graph(id='xg-vs-possession-away'),
    dcc.Graph(id='xg-vs-formation'),
    dcc.Graph(id='xg-vs-shots'),
    dcc.Graph(id='xg-vs-month'),
    dcc.Graph(id='xg-vs-time'),
])



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



if __name__ == '__main__':
    app.run_server(debug=True, port=8052)