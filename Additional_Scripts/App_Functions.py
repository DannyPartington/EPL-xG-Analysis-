import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Function to generate histogram of xG Error
def plot_xg_error_histogram(df):
    fig = px.histogram(df, x='xG - goals', nbins=20, title='Histogram of xG Error', color_discrete_sequence=['blue'])
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )
    return fig

# Function to generate heatmap of xG Error vs Total Goals
def plot_xg_error_vs_total_goals(df):
    fig = px.density_heatmap(df, x='Total Goals', y='xG - goals', title='xG Error vs Total Goals Heatmap',
                             color_continuous_scale='viridis')
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )
    return fig

# Function to generate xG league table
def calculate_xg_league_table(df):
    table = df.groupby('Team').agg({'xG - goals': 'sum'}).reset_index()
    table = table.sort_values(by='xG - goals', ascending=False)
    return table

# Function to style xG league table for Dash
def style_xg_league_table(table):
    return table.to_dict('records')

# Function to calculate xG to Goals ratio
def calculate_xg_to_goals(df):
    df['xG to Goals Ratio'] = df['Total xG'] / df['Total Goals']
    return df[['Team', 'Total Goals', 'Total xG', 'xG to Goals Ratio']].sort_values(by='xG to Goals Ratio', ascending=False)

# Function to style xG to Goals table for Dash
def style_xg_to_goals_table(df):
    return df.to_dict('records')

# Function to merge and rank xG statistics
def merge_and_rank_xg(avg_xG, avg_xGA):
    merged_df = avg_xG.merge(avg_xGA, on='Team', how='inner')
    return merged_df

# Function to style merged xG rankings
def style_xg_rankings(df):
    return df.to_dict('records')

# Function to generate home vs away xG scatter plot
def plot_home_vs_away_xg(avg_xG):
    fig = px.scatter(avg_xG, x='Avg_Home_xg', y='Avg_Away_xg', text='Team', title='Home vs Away xG')
    fig.update_traces(textposition='top center')
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )
    return fig

# Function to generate home vs away xGA scatter plot
def plot_home_vs_away_xGA(avg_xGA):
    fig = px.scatter(avg_xGA, x='Avg_Home_xGA', y='Avg_Away_xGA', text='Team', title='Home vs Away xGA', color_discrete_sequence=['red'])
    fig.update_traces(textposition='top center')
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )
    return fig

# Function to plot Premier League ranking vs xG ranking
def plot_pl_vs_xg_ranking(xg_goals_df, league_table):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xg_goals_df['Ranking'], y=xg_goals_df['xG to Goals Ratio'],
        mode='markers+text', text=xg_goals_df['Team'],
        marker=dict(size=10, color='cyan')
    ))
    fig.update_layout(
        title="PL Ranking vs xG to Goals Ranking",
        xaxis_title="League Ranking",
        yaxis_title="xG to Goals Ratio",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )
    return fig

# Function to generate xG summary table
def generate_xg_summary_table(df):
    summary = df.describe().transpose()
    return summary.to_dict('records')
