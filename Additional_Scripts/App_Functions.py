import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from dash import dcc, html, dash_table
from scipy.stats import gaussian_kde
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np


def plot_xg_error_histogram(df):
    """
    Plots a histogram with KDE for xG Error in Plotly, ensuring y-axis represents actual game counts,
    with clear bar edges.
    """

    # Extract xG error data
    xg_error = df['xG - goals'].dropna()

    # Define bin edges 
    bins = np.arange(xg_error.min(), xg_error.max() + 0.3, 0.3)

    # Create histogram with visible edges
    hist = go.Histogram(
        x=xg_error, 
        xbins=dict(start=bins.min(), end=bins.max(), size=0.3), 
        marker=dict(color='blue', line=dict(color='black', width=1)),  
        opacity=0.7,
        name="xG Error"
    )

    #Use NumPy KDE
    kde_x = np.linspace(xg_error.min(), xg_error.max(), 200)
    kde = gaussian_kde(xg_error, bw_method=0.3) 
    kde_y = kde(kde_x) * len(xg_error) * 0.3  

    kde_curve = go.Scatter(
        x=kde_x, y=kde_y,  
        mode='lines',
        line=dict(color='cyan', width=2),
        name="KDE"
    )

    # Combine both traces in figure
    fig = go.Figure([hist, kde_curve])

    # Update layout for styling
    fig.update_layout(
        title='Histogram with KDE for xG Error, where Error = xG - actual goals',
        xaxis_title='xG Error (xG - goals)',
        yaxis_title='Number of Games', 
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    return fig



def plot_xg_error_vs_total_goals(df):
    fig = px.density_heatmap(
        df, 
        x='Total Goals', 
        y='xG - goals', 
        nbinsx=18, 
        nbinsy=30, 
        color_continuous_scale=[(0, "black"), (0.02, "purple"), (0.3, "orange"), (1, "yellow")],
    )
    
    # Add faint dotted line at y = 0
    fig.add_shape(
        type='line',
        x0=df['Total Goals'].min(),  
        x1=df['Total Goals'].max(),  
        y0=0,
        y1=0,
        line=dict(
            color='white',
            width=1,
            dash='dot', 
        ),
        opacity=0.5  
    )
    
    fig.update_layout(
        title='Heatmap: Total Goals vs xG Error', 
        xaxis_title='Total Goals', 
        yaxis_title='xG Error (xG - Goals)',
        template='plotly_dark',
        plot_bgcolor='black',  
        paper_bgcolor='black',  
        font=dict(color='white'),
        coloraxis_colorbar=dict(title="No. of Games") 
    )
    
    return fig



def calculate_xg_league_table(df, league_table):
    """
    Generates an xG-based league table with rank changes by comparing xG-based ranking with actual league ranking.

    Args:
        df (pd.DataFrame): DataFrame containing match-level xG data with 'Home', 'Away', 'Home_xg', and 'Away_xg'.
        league_table (pd.DataFrame): DataFrame containing actual league rankings with 'Team' and 'Ranking' columns.

    Returns:
        list[dict]: A ranked xG-based league table formatted for Dash DataTable.
    """

    # Initialize dictionaries to hold xG stats
    team_xG_points = {}
    team_xG_scored = {}
    team_xG_conceded = {}

    # Process each fixture and allocate points and xG values
    for index, row in df.iterrows():
        home_team = row['Home']
        away_team = row['Away']
        home_xg = row['Home_xg']
        away_xg = row['Away_xg']

        # Ensure both teams are initialized in dictionaries
        for team in [home_team, away_team]:
            if team not in team_xG_points:
                team_xG_points[team] = 0
                team_xG_scored[team] = 0
                team_xG_conceded[team] = 0

        # Allocate points based on xG comparison
        if home_xg >= away_xg + 0.5:
            team_xG_points[home_team] += 3  # Home team wins
        elif away_xg >= home_xg + 0.5:
            team_xG_points[away_team] += 3  # Away team wins
        else:
            team_xG_points[home_team] += 1  # Draw
            team_xG_points[away_team] += 1

        # Update xG scored and conceded
        team_xG_scored[home_team] += home_xg
        team_xG_conceded[home_team] += away_xg
        team_xG_scored[away_team] += away_xg
        team_xG_conceded[away_team] += home_xg

    # Create a DataFrame for the xG-based league table
    xG_league_table = pd.DataFrame({
        'Team': team_xG_points.keys(),
        'Points': team_xG_points.values(),
        'Scored': [team_xG_scored[team] for team in team_xG_points.keys()],
        'Conceded': [team_xG_conceded[team] for team in team_xG_points.keys()]
    })

    #Round numerical columns to integers
    float_columns = ['Scored', 'Conceded']
    xG_league_table[float_columns] = xG_league_table[float_columns].round(0).astype(int)

    #Calculate xG Goal Difference
    xG_league_table['Goal_Difference'] = xG_league_table['Scored'] - xG_league_table['Conceded']

    #Rank teams based on xG points and goal difference
    xG_league_table = xG_league_table.sort_values(by=['Points', 'Goal_Difference'], ascending=[False, False]).reset_index(drop=True)
    xG_league_table['Rank'] = xG_league_table.index + 1

    #Merge with actual league rankings
    league_table = league_table[['Team', 'Ranking']] 
    xG_league_table = xG_league_table.merge(league_table, on='Team', how='left')

    #Calculate rank change
    xG_league_table['Rank Change'] = xG_league_table['Ranking'] - xG_league_table['Rank']

    #Format rank change for display
    xG_league_table['Rank'] = xG_league_table.apply(
        lambda row: f"{row['Rank']} (+{row['Rank Change']})" if row['Rank Change'] > 0 else 
                    f"{row['Rank']} ({row['Rank Change']})" if row['Rank Change'] < 0 else 
                    f"{row['Rank']} (0)",
        axis=1
    )

    #Rearrange columns
    xG_league_table = xG_league_table[['Rank', 'Team', 'Points', 'Scored', 'Conceded', 'Goal_Difference']]

    return xG_league_table.to_dict('records') 


def calculate_xg_to_goals(df, league_table):
    #Aggregate Home & Away xG and Goals
    home_xg_goals = df.groupby('Home')[['Home_xg', 'Home_Goals']].sum().reset_index()
    away_xg_goals = df.groupby('Away')[['Away_xg', 'Away_Goals']].sum().reset_index()
    
    #Merge home and away stats into a single DataFrame
    team_stats = home_xg_goals.merge(
        away_xg_goals,
        left_on='Home',
        right_on='Away',
        suffixes=('_home', '_away')
    )
    
    #Calculate Total xG and Goals for each team
    team_stats['Total_xG'] = team_stats['Home_xg'] + team_stats['Away_xg']
    team_stats['Total_Goals'] = team_stats['Home_Goals'] + team_stats['Away_Goals']
    
    #Calculate the xG:Goals ratio (handle division by zero)
    team_stats['xG_to_Goals_Ratio'] = team_stats['Total_xG'] / team_stats['Total_Goals']
    team_stats['xG_to_Goals_Ratio'] = team_stats['xG_to_Goals_Ratio'].replace([float('inf'), -float('inf')], None)
    #Sort teams by xG:Goals ratio (ascending)
    team_stats = team_stats.sort_values(by='xG_to_Goals_Ratio', ascending=True)
    
    #Rename "Home" column to "Team"
    team_stats = team_stats.rename(columns={'Home': 'Team'})

    #Merge with league_table to add only the "Ranking" column
    league_table = league_table[['Team', 'Ranking']]  
    team_stats = team_stats.merge(league_table, on='Team', how='left')  # Merge to add "Ranking"

    #Format & Round Numbers
    team_stats['Total_xG'] = team_stats['Total_xG'].astype(int)  # Convert xG to integer
    team_stats['xG_to_Goals_Ratio'] = team_stats['xG_to_Goals_Ratio'].map('{:.2f}'.format)  # Round xG ratio to 2 decimal places
    
    return team_stats[['Team', 'Total_Goals', 'Total_xG', 'xG_to_Goals_Ratio', 'Ranking']]  # Include "Ranking" in output



def style_xg_to_goals_table(df):
    return df.to_dict('records')


def plot_pl_vs_xg_ranking(xg_table):

    # Ensure rankings are sorted properly for visualization
    xg_table = xg_table.copy()
    xg_table = xg_table.sort_values(by='Ranking', ascending=False)
    
    xg_table['Ratio Ranking'] = xg_table.index + 1

    # Generate short team names (for Manchester clubs and others)
    xg_table['Short Team Name'] = xg_table['Team'].apply(
        lambda x: 'MC' if x == 'Manchester City' else 'MU' if x == 'Manchester Utd' else x[:3]
    )

    # Create a numeric Y-axis position for each team
    xg_table['Numeric Y'] = range(len(xg_table), 0, -1)

    # Initialize plotly figure
    fig = go.Figure()

    # Add league ranking scatter points (Blue)
    fig.add_trace(go.Scatter(
        x=xg_table['Ranking'],
        y=xg_table['Numeric Y'],
        mode='markers',
        marker=dict(color='blue', size=10),
        name='League Ranking'
    ))

    # Add xG ratio ranking scatter points (Orange for better readability)
    fig.add_trace(go.Scatter(
        x=xg_table['Ratio Ranking'],
        y=xg_table['Numeric Y'],
        mode='markers',
        marker=dict(color='orange', size=10),
        name='Ratio Ranking'
    ))

    # Add connecting lines (arrows)
    for _, row in xg_table.iterrows():
        line_color = 'green' if row['Ratio Ranking'] < row['Ranking'] else 'red'
        
        fig.add_trace(go.Scatter(
            x=[row['Ranking'], row['Ratio Ranking']],
            y=[row['Numeric Y'], row['Numeric Y']],
            mode='lines+markers',
            marker=dict(size=5, color=line_color),
            line=dict(color=line_color, width=2, dash='dash'),
            showlegend=False
        ))

    # Update y-axis labels (Match font with table on the left)
    fig.update_yaxes(
        tickvals=xg_table['Numeric Y'],
        ticktext=[f"({rank}) {team}" for rank, team in zip(xg_table['Ranking'], xg_table['Short Team Name'])],
        autorange='reversed',  # Ensure rank 20 is at the bottom
        tickfont=dict(family="Arial", size=14, color="white")  # Match font with table
    )

    # Update x-axis properties (Match font with table)
    fig.update_xaxes(
        title_text='Rank',
        tickmode='array',
        tickvals=list(range(1, 21, 1)),
        autorange='reversed',  
        title_font=dict(family="Arial", size=14, color="white"),  
        tickfont=dict(family="Arial", size=14, color="white")  
    )

    # Apply layout styling
    fig.update_layout(
        title_text='PL Ranking vs G:xG Ranking',
        title_font=dict(size=16, family="Arial", color="white"),
        template="plotly_dark",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", family="Arial"),
        legend=dict(
            title=None,  
            title_font=dict(color="white", size=14),  
            font=dict(color="white"), 
            bgcolor='black', 
            bordercolor='white', 
            borderwidth=1
        ),
        showlegend=True,  
        height=640,
        width=800
    )


    return fig



def plot_home_vs_away_xg(df):
    # Create a new column for formatted labels
    df['Short_Label'] = df['Team'].apply(lambda x: 
        'MC' if x == 'Manchester City' else
        'MU' if x == 'Manchester Utd' else
        x[:3].upper()
    )

    # Add ranking to the label
    df['Label'] = df['Short_Label'] + " (" + df['Ranking'].astype(str) + ")"

    # Define text positioning for MU
    df['Text_Position'] = df['Short_Label'].apply(lambda x: 'middle right' if x == 'MU' else 'top center')

    fig = px.scatter(df, x='Avg_Home_xg', y='Avg_Away_xg', text='Label', title='Home vs Away Average xG Created Per Game')
    fig.update_traces(textposition=df['Text_Position'], textfont=dict(size=10))   
    fig.update_layout(template='plotly_dark', plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
    return fig


def plot_home_vs_away_xGA(df):
    # Create a new column for formatted labels
    df['Short_Label'] = df['Team'].apply(lambda x: 
        'MC' if x == 'Manchester City' else
        'MU' if x == 'Manchester Utd' else
        x[:3].upper()
    )

    # Add ranking to the label
    df['Label'] = df['Short_Label'] + " (" + df['Ranking'].astype(str) + ")"

    # Initialize offsets
    df['x_offset'] = 0.0
    df['y_offset'] = 0.0

    # Apply custom positioning adjustments
    for index, row in df.iterrows():
        short_name = row['Short_Label']
        
        if short_name in ["CRY", "WOL", "BUR"]:
            df.at[index, 'x_offset'] = -0.03 
        elif short_name == "CHE":
            df.at[index, 'y_offset'] = -0.03 
        elif short_name == 'BOU':
            df.at[index, 'y_offset'] = 0.01  
        elif short_name == 'MU':
            df.at[index, 'x_offset'] = 0.02   

    fig = px.scatter(df, x='Avg_Home_xGA', y='Avg_Away_xGA', text='Label', title='Home vs Away Average xG Conceded Per Game', color_discrete_sequence=['red'])

    
    fig.update_traces(textposition='top center', textfont=dict(size=10))

    # Adjust text labels manually
    for i in range(len(df)):
        fig.data[0].x[i] += df.iloc[i]['x_offset']
        fig.data[0].y[i] += df.iloc[i]['y_offset']

    fig.update_layout(template='plotly_dark', plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
    return fig




def merge_and_rank_xg(avg_xg, avg_xGA):
    # Merge xG Created & xG Conceded DataFrames
   xg_combined = avg_xg.merge(avg_xGA, on=['Team', 'Ranking'], how='inner')

   # Assign Rankings for xG and xGA
   xg_combined['Rank_Home_xG'] = xg_combined['Avg_Home_xg'].rank(ascending=False, method='min')
   xg_combined['Rank_Away_xG'] = xg_combined['Avg_Away_xg'].rank(ascending=False, method='min')
   xg_combined['Rank_Home_xGA'] = xg_combined['Avg_Home_xGA'].rank(ascending=True, method='min')  
   xg_combined['Rank_Away_xGA'] = xg_combined['Avg_Away_xGA'].rank(ascending=True, method='min')  

   #Convert Rankings to Integers
   xg_combined[['Rank_Home_xG', 'Rank_Away_xG', 'Rank_Home_xGA', 'Rank_Away_xGA']] = xg_combined[
       ['Rank_Home_xG', 'Rank_Away_xG', 'Rank_Home_xGA', 'Rank_Away_xGA']
   ].astype(int)

   #Select Only Required Columns Before Styling
   xg_combined = xg_combined[['Ranking', 'Team', 'Rank_Home_xG', 'Rank_Away_xG', 'Rank_Home_xGA', 'Rank_Away_xGA']]

   #Sort the Table by Actual League Ranking
   xg_combined = xg_combined.sort_values(by='Ranking', ascending=True).reset_index(drop=True)

   return xg_combined  # Returning cleaned dataframe to be styled separately

def style_xg_rankings(df):
    return df.to_dict('records')

def generate_xg_summary_table(df):
    summary = df.describe().transpose()
    return summary.to_dict('records')


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
    
    #Set Black Background for Dark Mode
    fig_home.update_layout(title=f"{team} - Home xG vs Possession", xaxis_title="Possession (%)", yaxis_title="Net xG", 
                           template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black")
    fig_away.update_layout(title=f"{team} - Away xG vs Possession", xaxis_title="Possession (%)", yaxis_title="Net xG", 
                           template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black")
    
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
        template="plotly_dark",
        plot_bgcolor="black",
        paper_bgcolor="black"
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
    fig.update_layout(template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black")
    fig.update_traces(marker=dict(size=8, color="blue"))
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

    fig.add_hline(y=0, line=dict(color='black', width=1)) 

    fig.update_layout(title=f"{team} - Net xG by Match Time", xaxis_title="Match Period", yaxis_title="Average Net xG", template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black")
    return fig



def analyze_xg_vs_month(team, matches):
    team_matches = matches[(matches['Home'] == team) | (matches['Away'] == team)].copy()

    team_matches = team_matches.assign(
        xG_Created=np.where(team_matches['Home'] == team, team_matches['Home_xg'], team_matches['Away_xg']),
        xG_Conceded=np.where(team_matches['Home'] == team, team_matches['Away_xg'], team_matches['Home_xg'])
    )

    #Define correct month order (Premier League season runs August - May)
    month_order = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]

    #Convert full month names to abbreviations
    team_matches["Month"] = pd.to_datetime(team_matches["Date"]).dt.strftime("%b") 

    #Compute monthly averages
    monthly_xg_created = team_matches.groupby("Month")["xG_Created"].mean()
    monthly_xg_conceded = team_matches.groupby("Month")["xG_Conceded"].mean()

    #Sort by the predefined month order
    monthly_xg_created = monthly_xg_created.reindex(month_order)
    monthly_xg_conceded = monthly_xg_conceded.reindex(month_order)

    #Plot the xG trends over the season
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_xg_created.index, y=monthly_xg_created, mode='lines+markers', name="xG Created", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=monthly_xg_conceded.index, y=monthly_xg_conceded, mode='lines+markers', name="xG Conceded", line=dict(color="red")))

    #Set Black Background for Dark Mode
    fig.update_layout(
        title=f"{team} - xG Trend by Month",
        xaxis_title="Month",
        yaxis_title="Average xG",
        template="plotly_dark",
        plot_bgcolor="black",
        paper_bgcolor="black"
    )

    return fig



def generate_team_summary_table(team, xg_goal_ratio_table, league_table, xg_league_table, avg_xg_table, avg_xga_table):
    """
    Generates a summary table for the selected team with key performance metrics.
    """

    actual_rank = league_table.loc[league_table['Team'] == team, 'Ranking'].values[0]
    actual_points = league_table.loc[league_table['Team'] == team, 'Points'].values[0]

    xg_rank = xg_league_table.loc[xg_league_table['Team'] == team, 'Rank'].values[0]
    xg_points = xg_league_table.loc[xg_league_table['Team'] == team, 'Points'].values[0]

    xg_ratio = xg_goal_ratio_table.loc[xg_goal_ratio_table['Team'] == team, 'xG_to_Goals_Ratio'].values[0]

    avg_home_xg_created = avg_xg_table.loc[avg_xg_table['Team'] == team, 'Avg_Home_xg'].values[0]
    avg_home_xg_conceded = avg_xga_table.loc[avg_xga_table['Team'] == team, 'Avg_Home_xGA'].values[0]

    avg_away_xg_created = avg_xg_table.loc[avg_xg_table['Team'] == team, 'Avg_Away_xg'].values[0]
    avg_away_xg_conceded = avg_xga_table.loc[avg_xga_table['Team'] == team, 'Avg_Away_xGA'].values[0]

    #Create summary table DataFrame
    summary_df = pd.DataFrame({
        "23/24 Rank (Pts)": [f"{actual_rank} ({actual_points})"],
        "23/24 xG Rank (Pts)": [f"{xg_rank} ({xg_points})"],
        "xG to Goal Ratio": [f"{xg_ratio:.2f}"],
        "Avg xG (H)": [f"{avg_home_xg_created:.2f}"],
        "Avg xGA (H)": [f"{avg_home_xg_conceded:.2f}"],
        "Avg xG (A)": [f"{avg_away_xg_created:.2f}"],
        "Avg xGA (A)": [f"{avg_away_xg_conceded:.2f}"]
    })

    #Final DataTable with normal layout and only font customization
    summary_table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in summary_df.columns],
        data=summary_df.to_dict("records"),
        style_table={
            "width": "100%",
            "overflowX": "hidden",
        },
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "#343a40", 
            "color": "white",
            "textAlign": "center",
            "fontSize": "16px",  
            "padding": "10px"
        },
        style_cell={
            "textAlign": "center",
            "fontSize": "16px", 
            "color": "white",
            "backgroundColor": "black",
            "padding": "8px",  
            "whiteSpace": "normal",
        },
        style_data_conditional=[
            {'if': {'state': 'active'}, 'backgroundColor': '#333333'},
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#1e1e1e'},
        ]
    )

    return summary_table
