# -*- coding: utf-8 -*-



import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go



def plot_xg_error_histogram(df):
    """
    Plots a histogram with KDE for xG Error.
    
  
    Returns:
        None (Displays the plot)
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(df['xG - goals']))
    fig.update_layout(title='Histogram with KDE for xG Error')
    fig.update_layout(xaxis_title='xG Error')
    fig.update_layout(yaxis_title='Number of Games')
    return fig
    

def plot_xg_error_vs_total_goals(df):
    """
    Plots a heatmap showing the relationship between Total Goals and xG Error (xG - Goals),
    including a correlation calculation.

    Parameters:
        df (DataFrame): The dataset containing 'Total Goals' and 'xG - goals' columns.

    Returns:
        None (Displays the plot)
    """
    # Define custom bin edges for xG Error in increments of 1/3
    xg_error_bins = np.arange(df['xG - goals'].min(), df['xG - goals'].max() + (1/3), 1/3)

    # Calculate correlation between Total Goals and xG Error
    correlation = df['Total Goals'].corr(df['xG - goals'])

    fig = go.Figure())
    ax = sns.histplot(
        data=df,
        x='Total Goals',
        y='xG - goals',
        bins=[18, xg_error_bins],
        cmap='viridis',
        cbar=True
    )

    # Adjust the color bar to increments of 2
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.arange(0, cbar.vmax + 2, 2))

    # Add a horizontal dotted line at y = 0
    plt.axhline(y=0.03, color='black', linestyle='dotted', linewidth=1)

    # Add titles and labels
    fig.update_layout(title='Heatmap: Total Goals vs xG Error (xG - Goals)', fontsize=14)
    fig.update_layout(xaxis_title='Total Goals', fontsize=12)
    fig.update_layout(yaxis_title='xG Error (xG - Goals)', fontsize=12)

    # Display correlation figure on the chart
    plt.text(
        x=5,
        y=df['xG - goals'].max(),
        s=f"Correlation: {correlation:.2f}",
        fontsize=12,
        color='red',
        bbox=dict(facecolor='white', alpha=0.8)
    )
    return fig
    
    
def generate_xg_summary_table(df):
    """
    Generates a summary table of xG statistics grouped by Total Goals.
    
    Parameters:
        df (DataFrame): The dataset containing 'Total Goals' and 'xG - goals' columns.
    
    Returns:
        DataFrame: Summary table without an index.
    """
    # Group the data by Total Goals
    grouped = df.groupby('Total Goals')

    # Calculate expanded statistics
    summary_table = grouped.apply(lambda group: pd.Series({
        'Number of Games': int(len(group)),
        'Mean Absolute Error': round(group['xG - goals'].abs().mean(), 2),
        'xG Error Standard Deviation': round(group['xG - goals'].std(), 2),
        'Pct of Games |xG - Goals| > 1': (group['xG - goals'].abs() > 1).mean() * 100,
        'Underpredicted (%)': (group['xG - goals'] < 0).mean() * 100,  
        'Overpredicted (%)': (group['xG - goals'] > 0).mean() * 100, 
        'xG to Goal Ratio': round(group['Total xG'].sum() / group['Total Goals'].sum(), 3) if group['Total Goals'].sum() != 0 else 'inf'
    })).reset_index(drop=True)  # 🔹 Hides the index by resetting it and dropping the original one

    # Ensure integer formatting where appropriate
    summary_table['Number of Games'] = summary_table['Number of Games'].astype(int)
    summary_table['Pct of Games |xG - Goals| > 1'] = summary_table['Pct of Games |xG - Goals| > 1'].round().astype(int)
    summary_table['Underpredicted (%)'] = summary_table['Underpredicted (%)'].round().astype(int)
    summary_table['Overpredicted (%)'] = summary_table['Overpredicted (%)'].round().astype(int)

    return summary_table




def plot_home_vs_away_xg(avg_xg, shorten_team_name):
    """
    Creates a scatter plot of Home xG vs. Away xG per team with team labels.

    Parameters:
    - avg_xg (DataFrame): A DataFrame containing 'Avg_Home_xg', 'Avg_Away_xg', 'Team', and 'Ranking'.
    - shorten_team_name (function): A function to shorten team names.

    Returns:
    - None (displays the plot).
    """

    fig = go.Figure())
    fig = px.scatter(x=avg_xg['Avg_Home_xg'], y=avg_xg['Avg_Away_xg'])

    # Add text annotations with team abbreviations
    for i in range(len(avg_xg)):
        short_name = shorten_team_name(avg_xg['Team'][i])

        # Adjust label positions to avoid overlap
        vertical_offset = -0.02 if short_name == "MU" else 0.01 if short_name == "BRI" else 0

        plt.text(
            avg_xg['Avg_Home_xg'][i] + 0.02,
            avg_xg['Avg_Away_xg'][i] + vertical_offset,
            f"{short_name} ({avg_xg['Ranking'][i]})",
            fontsize=8, ha='left', va='center', fontweight='bold', color='black'
        )

    # Add labels and title
    fig.update_layout(xaxis_title='Average Home xG', fontsize=12, fontweight='bold')
    fig.update_layout(yaxis_title='Average Away xG', fontsize=12, fontweight='bold')
    fig.update_layout(title='Scatter Plot: Home xG vs. Away xG per Team', fontsize=14, fontweight='bold', color='#333')

    # Set x-axis increments to 0.2
    x_min, x_max = avg_xg['Avg_Home_xg'].min(), avg_xg['Avg_Home_xg'].max()
    plt.xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 0.2, 0.2), fontsize=12)
    plt.yticks(fontsize=12)

    # Add stylish grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show plot
    return fig
    

def plot_home_vs_away_xGA(avg_xGA, shorten_team_name):
    """
    Creates a scatter plot of Home xGA vs. Away xGA per team with team labels.

    Parameters:
    - avg_xGA (DataFrame): A DataFrame containing 'Avg_Home_xGA', 'Avg_Away_xGA', 'Team', and 'Ranking'.
    - shorten_team_name (function): A function to shorten team names.

    Returns:
    - None (displays the plot).
    """

    fig = go.Figure())

    plt.scatter(
        avg_xGA['Avg_Home_xGA'], avg_xGA['Avg_Away_xGA'],
        s=100, c='red', alpha=0.6, edgecolors='w', marker='o'
    )

    # Add text annotations with specific positioning adjustments
    for i in range(len(avg_xGA)):
        short_name = shorten_team_name(avg_xGA['Team'][i])  # Apply abbreviation function

        # **Default: Labels to the right**
        x_offset, y_offset = 0.015, 0  # Shift text to the right

        # **Special cases**
        if short_name in ["CRY", "WOL", "BUR"]:  
            x_offset = -0.015  # Shift text to the left
        elif short_name == "CHE":
            y_offset = -0.01  # Drop slightly downward
        elif short_name == 'BOU':
            y_offset = 0.01

        plt.text(
            avg_xGA['Avg_Home_xGA'][i] + x_offset,  
            avg_xGA['Avg_Away_xGA'][i] + y_offset,  
            f"{short_name} ({avg_xGA['Ranking'][i]})",  
            fontsize=8, ha='left' if x_offset > 0 else 'right', va='center', fontweight='bold', color='black'
        ) 

    # **Set x and y axis range & increments**
    x_min, x_max = 0.5, np.ceil(avg_xGA['Avg_Home_xGA'].max())  
    y_min, y_max = 0.5, np.ceil(avg_xGA['Avg_Away_xGA'].max())

    plt.xticks(np.arange(x_min, x_max + 0.25, 0.25), fontsize=12)
    plt.yticks(np.arange(y_min, y_max + 0.25, 0.25), fontsize=12)

    # Add labels and title
    fig.update_layout(xaxis_title='Average Home xGA', fontsize=12, fontweight='bold')
    fig.update_layout(yaxis_title='Average Away xGA', fontsize=12, fontweight='bold')
    fig.update_layout(title='Scatter Plot: Home xGA vs. Away xGA per Team', fontsize=14, fontweight='bold', color='#333')

    # Add stylish grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show plot
    return fig




def merge_and_rank_xg(avg_xg, avg_xGA):
    """
    Merges xG Created & xG Conceded DataFrames, calculates rankings, and prepares a styled summary table.

    Parameters:
    - avg_xg (DataFrame): DataFrame containing xG Created metrics.
    - avg_xGA (DataFrame): DataFrame containing xGA Conceded metrics.

    Returns:
    - styled_xg_combined (Styler): A styled Pandas DataFrame.
    """

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


def style_xg_rankings(data):
    """
    Styles the xG rankings DataFrame for better visualization.

    Parameters:
    - data (DataFrame): A cleaned xG rankings DataFrame.

    Returns:
    - styled (Styler): A styled Pandas Styler object.
    """

    styles = pd.DataFrame('font-size: 16px; text-align: center', index=data.index, columns=data.columns)  # Set default font size & center text
    
    for index, row in data.iterrows():
        for col in ['Rank_Home_xG', 'Rank_Away_xG', 'Rank_Home_xGA', 'Rank_Away_xGA']:
            if row[col] < row['Ranking']:
                styles.at[index, col] += '; color: green; font-weight: bold'  # Better ranking
            elif row[col] > row['Ranking']:
                styles.at[index, col] += '; color: red; font-weight: bold'  # Worse ranking
    
    # ✅ Make team names bold
    styles['Team'] = 'font-weight: bold; font-size: 16px; text-align: left'
    
    styled = data.style.apply(lambda x: styles, axis=None)  # Apply styles

    # ✅ Center-align column headers
    styled.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]}
    ])

    return styled


def calculate_xg_to_goals(df):
    """
    Calculates the total xG, total goals, and xG-to-goals ratio for each team.

    Parameters:
    - df (DataFrame): Match-level dataset containing Home/Away xG and Goals.

    Returns:
    - team_stats (DataFrame): A sorted DataFrame containing team-level xG statistics.
    """

    # ✅ Step 1: Aggregate Home & Away xG and Goals
    home_xg_goals = df.groupby('Home')[['Home_xg', 'Home_Goals']].sum().reset_index()
    away_xg_goals = df.groupby('Away')[['Away_xg', 'Away_Goals']].sum().reset_index()

    # ✅ Step 2: Merge home and away stats into a single DataFrame
    team_stats = home_xg_goals.merge(
        away_xg_goals,
        left_on='Home',
        right_on='Away',
        suffixes=('_home', '_away')
    )

    # ✅ Step 3: Calculate Total xG and Goals for each team
    team_stats['Total_xG'] = team_stats['Home_xg'] + team_stats['Away_xg']
    team_stats['Total_Goals'] = team_stats['Home_Goals'] + team_stats['Away_Goals']

    # ✅ Step 4: Calculate the xG:Goals ratio (handle division by zero)
    team_stats['xG_to_Goals_Ratio'] = team_stats['Total_xG'] / team_stats['Total_Goals']
    team_stats['xG_to_Goals_Ratio'] = team_stats['xG_to_Goals_Ratio'].replace([float('inf'), -float('inf')], None)  # Replace infinite values

    # ✅ Step 5: Sort teams by xG:Goals ratio (ascending)
    team_stats = team_stats.sort_values(by='xG_to_Goals_Ratio', ascending=True)

    # ✅ Step 6: Format & Rename Columns
    team_stats = team_stats.rename(columns={'Home': 'Team'})  # Rename column
    team_stats['Total_xG'] = team_stats['Total_xG'].astype(int)  # Convert xG to integer
    team_stats['xG_to_Goals_Ratio'] = team_stats['xG_to_Goals_Ratio'].map('{:.2f}'.format)  # Round xG ratio to 2 decimal places

    return team_stats[['Team', 'Total_Goals', 'Total_xG', 'xG_to_Goals_Ratio']]  # Return only required columns


def style_xg_to_goals_table(data):
    """
    Styles the xG to Goals ratio table.

    Parameters:
    - data (DataFrame): A processed xG-to-goals ratio DataFrame.

    Returns:
    - styled (Styler): A styled Pandas DataFrame.
    """

    styled = data.style.set_properties(**{
        'text-align': 'center',
        'font-size': '16px'
    }).hide(axis="index")  # Hide index

    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
    ])

    return styled



def plot_pl_vs_xg_ranking(xg_goal_table, league_table):
    """
    Creates a dot plot comparing Premier League rankings with xG to Goals ratio rankings.

    Args:
        xg_goal_table (pd.DataFrame): DataFrame containing xG conversion statistics.
        league_table (pd.DataFrame): DataFrame containing actual Premier League rankings.

    Returns:
        None (Displays the plot)
    """

    # ✅ Merge xG conversion stats with actual league rankings
    team_ratio_stats = xg_goal_table.merge(
        league_table[['Team', 'Ranking']],
        on='Team'
    )

    # ✅ Generate a ranking for xG to Goals ratio
    team_ratio_stats['Ratio Ranking'] = team_ratio_stats.index + 1

    # ✅ Calculate the correlation between league ranking and xG conversion ranking
    correlation = team_ratio_stats['Ranking'].corr(team_ratio_stats['Ratio Ranking'])

    # ✅ Sort the table by actual league ranking (highest first)
    team_ratio_stats = team_ratio_stats.sort_values(by='Ranking', ascending=False)

    # ✅ Assign short team names for display
    team_ratio_stats['Short Team Name'] = team_ratio_stats['Team'].apply(
        lambda x: 'MC' if x == 'Manchester City' else 'MU' if x == 'Manchester Utd' else x[:3]
    )

    # ✅ Numeric Y-axis for ranking visualization
    team_ratio_stats['Numeric Y'] = range(len(team_ratio_stats), 0, -1)

    # ✅ Create the dot plot
    fig = go.Figure())

    # ✅ Draw arrows indicating changes between rankings
    for _, row in team_ratio_stats.iterrows():
        line_color = 'green' if row['Ratio Ranking'] < row['Ranking'] else 'red'
        dx = row['Ratio Ranking'] - row['Ranking']
        plt.arrow(row['Ranking'], row['Numeric Y'], dx, 0, 
                  color=line_color, alpha=0.6, width=0.1, 
                  head_width=0.3, head_length=0.5, length_includes_head=True)

    # ✅ Scatter plot for league rankings
    plt.scatter(team_ratio_stats['Ranking'], team_ratio_stats['Numeric Y'], 
                color='blue', label='League Ranking', s=100)

    # ✅ Scatter plot for xG conversion rankings
    plt.scatter(team_ratio_stats['Ratio Ranking'], team_ratio_stats['Numeric Y'], 
                color='black', label='Ratio Ranking', s=100)

    # ✅ Update y-axis labels with team names
    plt.yticks(
        ticks=team_ratio_stats['Numeric Y'], 
        labels=[f"({rank}) {team}" for rank, team in zip(team_ratio_stats['Ranking'], team_ratio_stats['Short Team Name'])],
        fontweight='bold'
    )

    # ✅ Add labels, title, and legend
    fig.update_layout(xaxis_title='Rank', fontweight='bold')
    fig.update_layout(title='PL Ranking vs G:xG Ranking', fontweight='bold')

    # ✅ Display correlation on the chart
    plt.text(
        0.02, 0.85, 
        f'Correlation: {correlation:.2f}',  
        transform=plt.gca().transAxes,  
        fontsize=12, fontweight='bold', color='black',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, boxstyle='round,pad=0.4')
    )

    plt.rcParams['font.weight'] = 'bold'
    plt.legend()
    plt.gca().set_facecolor('#D3D3D3')  # Dark gray background
    plt.gca().invert_yaxis()  # Rank 20 at the bottom
    plt.gca().invert_xaxis()  # Rank 1 on the left
    plt.xticks(ticks=range(1, 21, 1))  # Ensure x-axis ticks increase in whole numbers

    plt.grid(True, linestyle='--', alpha=0.5)

    # ✅ Show the plot
    return fig


def calculate_xg_league_table(df):
    """
    Generates an xG-based league table by allocating points and ranking teams based on expected goals.

    Args:
        df (pd.DataFrame): DataFrame containing match-level xG data with 'Home', 'Away', 'Home_xg', and 'Away_xg'.

    Returns:
        pd.DataFrame: A ranked xG-based league table.
    """

    # ✅ Initialize dictionaries to hold xG stats
    team_xG_points = {}
    team_xG_scored = {}
    team_xG_conceded = {}

    # ✅ Process each fixture and allocate points and xG values
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

    # ✅ Create a DataFrame for the xG-based league table
    xG_league_table = pd.DataFrame({
        'Team': team_xG_points.keys(),
        'Points': team_xG_points.values(),
        'Scored': [team_xG_scored[team] for team in team_xG_points.keys()],
        'Conceded': [team_xG_conceded[team] for team in team_xG_points.keys()]
    })

    # ✅ Round numerical columns to integers
    float_columns = ['Scored', 'Conceded']
    xG_league_table[float_columns] = xG_league_table[float_columns].round(0).astype(int)

    # ✅ Calculate xG Goal Difference
    xG_league_table['Goal_Difference'] = xG_league_table['Scored'] - xG_league_table['Conceded']

    # ✅ Rank teams based on xG points and goal difference
    xG_league_table = xG_league_table.sort_values(by=['Points', 'Goal_Difference'], ascending=[False, False]).reset_index(drop=True)
    xG_league_table['Rank'] = xG_league_table.index + 1

    # ✅ Rearrange columns
    xG_league_table = xG_league_table[['Rank', 'Team', 'Points', 'Scored', 'Conceded', 'Goal_Difference']]

    return xG_league_table


def style_xg_league_table(data):
    """
    Styles the xG-based league table for improved visualization.

    Args:
        data (pd.DataFrame): A processed xG league table.

    Returns:
        pd.Styler: A styled Pandas DataFrame.
    """

    styled = data.style.set_properties(**{
        'text-align': 'center',
        'font-size': '16px'
    }).hide(axis="index")  # Hide index

    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
    ])

    return styled



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
