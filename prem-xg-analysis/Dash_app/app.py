# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:34:44 2025

@author: d_par
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Prepare the data
team_data = pd.read_csv("Match_Stats_with_possession.csv")

# Add a function to calculate possession vs Net xG for a team
def generate_chart(team, venue, matches):
    # Filter data based on venue
    if venue == 'Home':
        team_matches = matches[matches['Home'] == team].copy()
        team_matches['Possession'] = team_matches['Home poss']
        team_matches['Net xG'] = team_matches['Home_xg'] - team_matches['Away_xg']
        team_matches['Opponent'] = team_matches['Away']
    elif venue == 'Away':
        team_matches = matches[matches['Away'] == team].copy()
        team_matches['Possession'] = team_matches['Away poss']
        team_matches['Net xG'] = team_matches['Away_xg'] - team_matches['Home_xg']
        team_matches['Opponent'] = team_matches['Home']
    else:
        raise ValueError("Venue must be 'Home' or 'Away'.")

    # Polynomial regression for the best-fit curve
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(team_matches[['Possession']])
    model = LinearRegression().fit(X_poly, team_matches['Net xG'])

    # Generate predictions
    x_range = np.linspace(team_matches['Possession'].min(), team_matches['Possession'].max(), 100)
    y_pred = model.predict(poly.transform(x_range.reshape(-1, 1)))

    # Create the plot using Plotly
    fig = px.scatter(team_matches, x='Possession', y='Net xG', text=team_matches['Opponent'].apply(lambda x: x[:3].upper()))
    fig.update_traces(textposition='top center')
    fig.add_scatter(x=x_range, y=y_pred, mode='lines', name='Best Fit Curve', line=dict(color='red'))

    # Add vertical line for optimal possession
    optimal_possession = x_range[np.argmax(y_pred)]
    fig.add_vline(x=optimal_possession, line_dash="dash", line_color="green", annotation_text=f"Optimal: {optimal_possession:.1f}%")

    # Update layout
    fig.update_layout(
        title=f"{team} ({venue}) - Possession vs Net xG",
        xaxis_title="Possession (%)",
        yaxis_title="Net xG (xG Created - xG Conceded)",
        legend_title="Legend",
    )
    return fig

# Dash app setup
app = dash.Dash(__name__)

# Dropdown options
teams = team_data['Home'].unique()
team_options = [{'label': team, 'value': team} for team in teams]

# Layout of the app
app.layout = html.Div([
    html.H1("Possession vs Net xG Analysis"),
    dcc.Dropdown(
        id='team-dropdown',
        options=team_options,
        value=teams[0],  # Default to the first team
        placeholder="Select a team"
    ),
    html.Div(id='home-chart-container', children=[]),
    html.Div(id='away-chart-container', children=[])
])

# Callback to update charts
@app.callback(
    [Output('home-chart-container', 'children'),
     Output('away-chart-container', 'children')],
    [Input('team-dropdown', 'value')]
)
def update_charts(selected_team):
    # Generate the home and away charts for the selected team
    home_chart = generate_chart(selected_team, 'Home', team_data)
    away_chart = generate_chart(selected_team, 'Away', team_data)

    # Return the charts as Plotly graphs
    return (
        dcc.Graph(figure=home_chart),
        dcc.Graph(figure=away_chart)
    )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)