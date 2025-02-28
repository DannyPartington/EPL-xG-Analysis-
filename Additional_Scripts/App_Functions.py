import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def plot_xg_error_histogram(df):
    fig = px.histogram(df, x='xG - goals', nbins=30, marginal='rug', color_discrete_sequence=['blue'])
    fig.update_layout(title='Histogram with KDE for xG Error', 
                      xaxis_title='xG Error', yaxis_title='Frequency', 
                      template='plotly_dark', plot_bgcolor='black', paper_bgcolor='black', 
                      font=dict(color='white'))
    return fig

def plot_xg_error_vs_total_goals(df):
    fig = px.density_heatmap(df, x='Total Goals', y='xG - goals', 
                             nbinsx=18, nbinsy=30, color_continuous_scale='viridis')
    fig.update_layout(title='Heatmap: Total Goals vs xG Error', 
                      xaxis_title='Total Goals', yaxis_title='xG Error (xG - Goals)',
                      template='plotly_dark', plot_bgcolor='black', paper_bgcolor='black',
                      font=dict(color='white'))
    return fig

def calculate_xg_league_table(df):
    home_stats = df.groupby('Home').agg({'xG - goals': 'sum'}).reset_index().rename(columns={'Home': 'Team'})
    away_stats = df.groupby('Away').agg({'xG - goals': 'sum'}).reset_index().rename(columns={'Away': 'Team'})
    team_stats = pd.concat([home_stats, away_stats]).groupby('Team').sum().reset_index()
    return team_stats

def style_xg_league_table(df):
    return df.to_dict('records')

def calculate_xg_to_goals(df):
    df['xG to Goals Ratio'] = df['Total xG'] / df['Total Goals'].replace(0, 1)
    return df[['Home', 'Total Goals', 'Total xG', 'xG to Goals Ratio']].rename(columns={'Home': 'Team'}).sort_values(by='xG to Goals Ratio', ascending=False)

def style_xg_to_goals_table(df):
    return df.to_dict('records')

def plot_home_vs_away_xg(df):
    fig = px.scatter(df, x='Avg_Home_xg', y='Avg_Away_xg', text='Team', title='Home vs Away xG')
    fig.update_traces(textposition='top center')
    fig.update_layout(template='plotly_dark', plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
    return fig

def plot_home_vs_away_xGA(df):
    fig = px.scatter(df, x='Avg_Home_xGA', y='Avg_Away_xGA', text='Team', title='Home vs Away xGA', color_discrete_sequence=['red'])
    fig.update_traces(textposition='top center')
    fig.update_layout(template='plotly_dark', plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
    return fig

def plot_pl_vs_xg_ranking(xg_table, league_table):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=league_table['Home Rank'], y=xg_table['xG to Goals Ratio'],
                              mode='markers+text', text=xg_table['Team'],
                              textposition='top center', marker=dict(color='blue')))
    fig.update_layout(title='PL Ranking vs G:xG Ranking',
                      xaxis_title='League Rank', yaxis_title='xG to Goals Ratio',
                      template='plotly_dark', plot_bgcolor='black', paper_bgcolor='black',
                      font=dict(color='white'))
    return fig

def merge_and_rank_xg(avg_xg, avg_xGA):
    # ✅ Step 1: Merge xG Created & xG Conceded DataFrames
   xg_combined = avg_xg.merge(avg_xGA, on=['Team', 'Ranking'], how='inner')

   # ✅ Step 2: Assign Rankings for xG and xGA
   xg_combined['Rank_Home_xG'] = xg_combined['Avg_Home_xg'].rank(ascending=False, method='min')
   xg_combined['Rank_Away_xG'] = xg_combined['Avg_Away_xg'].rank(ascending=False, method='min')
   xg_combined['Rank_Home_xGA'] = xg_combined['Avg_Home_xGA'].rank(ascending=True, method='min')  # Lower is better
   xg_combined['Rank_Away_xGA'] = xg_combined['Avg_Away_xGA'].rank(ascending=True, method='min')  # Lower is better

   # ✅ Step 3: Convert Rankings to Integers
   xg_combined[['Rank_Home_xG', 'Rank_Away_xG', 'Rank_Home_xGA', 'Rank_Away_xGA']] = xg_combined[
       ['Rank_Home_xG', 'Rank_Away_xG', 'Rank_Home_xGA', 'Rank_Away_xGA']
   ].astype(int)

   # ✅ Step 4: Select Only Required Columns Before Styling
   xg_combined = xg_combined[['Ranking', 'Team', 'Rank_Home_xG', 'Rank_Away_xG', 'Rank_Home_xGA', 'Rank_Away_xGA']]

   # ✅ Step 5: Sort the Table by Actual League Ranking
   xg_combined = xg_combined.sort_values(by='Ranking', ascending=True).reset_index(drop=True)

   return xg_combined  # Returning cleaned dataframe to be styled separately

def style_xg_rankings(df):
    return df.to_dict('records')

def generate_xg_summary_table(df):
    summary = df.describe().transpose()
    return summary.to_dict('records')
