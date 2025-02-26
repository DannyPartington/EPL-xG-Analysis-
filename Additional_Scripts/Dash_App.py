import dash
import os
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from Additional_Scripts.App_Functions import (
    plot_xg_error_histogram, plot_xg_error_vs_total_goals,
    calculate_xg_league_table, calculate_xg_to_goals, style_xg_to_goals_table,
    merge_and_rank_xg, style_xg_rankings,
    plot_home_vs_away_xg, plot_home_vs_away_xGA, plot_pl_vs_xg_ranking,
    style_xg_league_table, generate_xg_summary_table
)

# Initialize Dash app with Bootstrap (Dark Theme)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Get the absolute path to the directory containing Dash_App.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths for data files
file_path1 = os.path.join(BASE_DIR, '..', 'Data', 'team_data.csv')
file_path2 = os.path.join(BASE_DIR, '..', 'Data', 'cleaned_prem_data.csv')
file_path3 = os.path.join(BASE_DIR, '..', 'Data', 'avg_xg_table.csv')
file_path4 = os.path.join(BASE_DIR, '..', 'Data', 'avg_xga_table.csv')
file_path5 = os.path.join(BASE_DIR, '..', 'Data', 'Rankings.csv')

# Load data
team_data = pd.read_csv(file_path1)
cleaned_prem_data = pd.read_csv(file_path2)
avg_xG = pd.read_csv(file_path3)
avg_xGA = pd.read_csv(file_path4)
league_table = pd.read_csv(file_path5)

# Define Tabs
app.layout = dbc.Container([
    dcc.Tabs(id="tabs", value='team-xg-analysis', children=[
        dcc.Tab(label='Team xG Analysis', value='team-xg-analysis', style={'backgroundColor': '#1e1e1e', 'color': 'white'}),
        dcc.Tab(label='xG Error', value='xg-error', style={'backgroundColor': '#1e1e1e', 'color': 'white'}),
        dcc.Tab(label='xG League Table', value='xg-league-table', style={'backgroundColor': '#1e1e1e', 'color': 'white'}),
        dcc.Tab(label='League xG Analysis', value='league-xg-analysis', style={'backgroundColor': '#1e1e1e', 'color': 'white'})
    ], colors={"border": "#17a2b8", "primary": "#17a2b8", "background": "#121212"}),
    
    html.Div(id='page-content')
], fluid=True, style={'backgroundColor': '#121212', 'padding': '20px'})

# Define Callbacks to Load Pages
@app.callback(
    Output('page-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    print(f"Tab clicked: {tab}")  # Debug which tab is clicked
    
    if tab == 'xg-error':
        print("Loading xG Error tab...")
        return dbc.Container([
            html.H1("xG Error Analysis", style={'textAlign': 'center', 'color': '#17a2b8'}),
            dcc.Graph(figure=plot_xg_error_histogram(cleaned_prem_data)),
            dcc.Graph(figure=plot_xg_error_vs_total_goals(cleaned_prem_data))
        ])
    
    elif tab == 'xg-league-table':
        print("Loading xG League Table tab...")
        return dbc.Container([
            html.H1("xG League Table", style={'textAlign': 'center', 'color': '#17a2b8'}),
            dash_table.DataTable(style_xg_league_table(calculate_xg_league_table(cleaned_prem_data)))
        ])
    
    elif tab == 'league-xg-analysis':
        print("Loading League xG Analysis tab...")
        return dbc.Container([
            html.H1("League-Wide xG Analysis", style={'textAlign': 'center', 'color': '#17a2b8'}),
            dcc.Graph(figure=plot_home_vs_away_xg(avg_xG)),
            dcc.Graph(figure=plot_home_vs_away_xGA(avg_xGA)),
            dcc.Graph(figure=plot_pl_vs_xg_ranking(calculate_xg_to_goals(cleaned_prem_data), league_table)),
            dash_table.DataTable(merge_and_rank_xg(avg_xG, avg_xGA)),
            dash_table.DataTable(style_xg_rankings(merge_and_rank_xg(avg_xG, avg_xGA)))
        ])
    
    else:
        print("Loading Team xG Analysis tab...")
        return dbc.Container([
            html.H1("Team xG Analysis", style={'textAlign': 'center', 'color': '#17a2b8'}),
            dash_table.DataTable(generate_xg_summary_table(cleaned_prem_data))
        ])


server = app.server

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
