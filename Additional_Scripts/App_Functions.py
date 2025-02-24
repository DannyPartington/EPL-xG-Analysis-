# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:37:54 2025

@author: d_par

"""

from Team_Name_Formatting import standardize_team_name
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings


# Function to analyze xG vs Possession (Separate Home & Away Charts)
def analyze_xg_vs_possession(team, matches):
    team_matches = matches[(matches['Home'] == team) | (matches['Away'] == team)].copy()
    
    # Separate Home and Away Matches
    home_matches = team_matches[team_matches['Home'] == team].copy()
    away_matches = team_matches[team_matches['Away'] == team].copy()
    
    home_matches = home_matches.assign(Possession=home_matches['Home Possession'], Opponent=home_matches['Away'], Venue='H', Net_xG=home_matches['Home_xg'] - home_matches['Away_xg'])
    away_matches = away_matches.assign(Possession=away_matches['Away Possession'], Opponent=away_matches['Home'], Venue='A', Net_xG=away_matches['Away_xg'] - away_matches['Home_xg'])
    
    fig_home = go.Figure()
    fig_away = go.Figure()

    # Quadratic regression function
    def quadratic_fit(data, fig, color, label):
        X = data['Possession'].values.reshape(-1, 1)
        y = data['Net_xG'].astype(float).values
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        x_range = np.linspace(X.min(), X.max(), 100)
        y_pred = model.predict(poly.transform(x_range.reshape(-1, 1)))
        
        fig.add_trace(go.Scatter(
            x=data['Possession'], y=data['Net_xG'], mode='markers',
            marker=dict(color=color), name=label,
            hovertemplate="Opponent: %{text} (%{customdata})",
            text=data['Opponent'], customdata=data['Venue']
        ))
        fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', line=dict(color=color), name=f'{label} Trend'))

    quadratic_fit(home_matches, fig_home, 'blue', 'Home')
    quadratic_fit(away_matches, fig_away, 'red', 'Away')

    fig_home.update_layout(title=f"{team} - Home xG vs Possession", xaxis_title="Possession (%)", yaxis_title="Net xG", template="plotly_white")
    fig_away.update_layout(title=f"{team} - Away xG vs Possession", xaxis_title="Possession (%)", yaxis_title="Net xG", template="plotly_white")

    return fig_home, fig_away





# Function to analyze xG vs Formation 
def analyze_xg_vs_formation(team, matches):
    team_matches = matches[(matches['Home'] == team) | (matches['Away'] == team)].copy()

    team_matches = team_matches.assign(
        Formation=np.where(team_matches['Home'] == team, team_matches['Home Formation'], team_matches['Away Formation']),
        xG_Created=np.where(team_matches['Home'] == team, team_matches['Home_xg'], team_matches['Away_xg']),
        xG_Conceded=np.where(team_matches['Home'] == team, team_matches['Away_xg'], team_matches['Home_xg'])
    )

    formation_counts = team_matches['Formation'].value_counts()
    formation_xg_created = team_matches.groupby("Formation")["xG_Created"].mean()
    formation_xg_conceded = team_matches.groupby("Formation")["xG_Conceded"].mean()

    formatted_labels = [f"{f} ({formation_counts[f]})" for f in formation_xg_created.index]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=formatted_labels, y=formation_xg_created, name="xG Created", marker_color="green"))
    fig.add_trace(go.Bar(x=formatted_labels, y=formation_xg_conceded, name="xG Conceded", marker_color="red"))
    
    fig.update_layout(
        barmode='group', 
        title=f"{team} - xG by Formation",
        xaxis_title="Formation (Games Played)",
        yaxis_title="Average xG",
        template="plotly_white"
    )
    
    return fig




# Function to analyze xG vs Shots 
def analyze_xg_vs_shots(team, matches):
    team_matches = matches[(matches['Home'] == team) | (matches['Away'] == team)].copy()

    team_matches = team_matches.assign(
        Shots=np.where(team_matches['Home'] == team, team_matches['Home Shots'], team_matches['Away Shots']),
        xG=np.where(team_matches['Home'] == team, team_matches['Home_xg'], team_matches['Away_xg']),
        Opponent=np.where(team_matches['Home'] == team, team_matches['Away'], team_matches['Home']),
        Venue=np.where(team_matches['Home'] == team, 'H', 'A'),
        Match_Date=pd.to_datetime(team_matches['Date']).dt.strftime('%m/%d')
    )

    fig = px.scatter(
        team_matches, x="Shots", y="xG", 
        hover_data={"Shots": False, "xG": False, "Opponent": True, "Venue": True, "Match_Date": True},
        title=f"{team} - xG vs Shots Taken",
        labels={"Shots": "Shots Taken", "xG": "xG"}
    )

    fig.update_traces(marker=dict(size=8, color="blue"))
    return fig 



# Function to analyze xG vs Month (Line Chart for xG Created & Conceded)
def analyze_xg_vs_month(team, matches):
    team_matches = matches[(matches['Home'] == team) | (matches['Away'] == team)].copy()

    team_matches = team_matches.assign(
        xG_Created=np.where(team_matches['Home'] == team, team_matches['Home_xg'], team_matches['Away_xg']),
        xG_Conceded=np.where(team_matches['Home'] == team, team_matches['Away_xg'], team_matches['Home_xg'])
    )

    monthly_xg_created = team_matches.groupby("Month")["xG_Created"].mean()
    monthly_xg_conceded = team_matches.groupby("Month")["xG_Conceded"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_xg_created.index, y=monthly_xg_created, mode='lines+markers', name="xG Created", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=monthly_xg_conceded.index, y=monthly_xg_conceded, mode='lines+markers', name="xG Conceded", line=dict(color="red")))

    fig.update_layout(title=f"{team} - xG Trend by Month", xaxis_title="Month", yaxis_title="Average xG", template="plotly_white")
    return fig




# Function to analyze xG vs Time of Match (Bar Chart for Net xG with Game Count and y=0 line)
def analyze_xg_vs_time(team, matches):
    team_matches = matches[(matches['Home'] == team) | (matches['Away'] == team)].copy()

    team_matches = team_matches.assign(
        Net_xG=np.where(team_matches['Home'] == team, team_matches['Home_xg'] - team_matches['Away_xg'], team_matches['Away_xg'] - team_matches['Home_xg'])
    )

    xg_by_time = team_matches.groupby("Match Period")["Net_xG"].mean()
    match_counts = team_matches["Match Period"].value_counts()

    labels = [f"{period} ({match_counts[period]})" for period in xg_by_time.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=xg_by_time.values, marker_color=["blue", "red"], name="Net xG"))

    fig.add_hline(y=0, line=dict(color='black', width=1))  # Add y=0 reference line

    fig.update_layout(title=f"{team} - Net xG by Match Time", xaxis_title="Match Period", yaxis_title="Average Net xG", template="plotly_white")
    return fig
