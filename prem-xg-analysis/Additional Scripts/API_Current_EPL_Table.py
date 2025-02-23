# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:04:14 2025

@author: d_par
"""

import requests
import pandas as pd

# API endpoint for EPL Standings
url = "https://api.football-data.org/v4/competitions/PL/standings"

# Your API Key (get from football-data.org)
headers = {
    "X-Auth-Token": "59194b9dba8a42a8a29957d443a52548"
}

# Make the request
response = requests.get(url, headers=headers)
data = response.json()

# Extract standings
standings = data["standings"][0]["table"]

# Convert to DataFrame
df = pd.DataFrame([
    {
        "Position": team["position"],
        "Team": team["team"]["name"],
        "Played": team["playedGames"],
        "Points": team["points"],
        "Goal Difference": team["goalDifference"]
    }
    for team in standings
])


# Define manual mapping for corrections
team_name_mapping = {
    "Liverpool FC" : "Liverpool",
    "Arsenal FC" : "Arsenal",
   "Nottingham Forest FC" : "Nott'ham Forest",
    "Manchester City FC" : "Manchester City",
    "Newcastle United FC" : "Newcastle Utd",
    "Chelsea FC" : "Chelsea",
    "AFC Bournemouth" : "Bournemouth",
    "Aston Villa FC" : "Aston Villa",
    "Fulham FC" : "Fulham",
    "Brighton & Hove Albion FC" : "Brighton",
    "Brentford FC" : "Brentford",
    "Crystal Palace FC" : "Crystal Palace",
    "Manchester United FC" : "Manchester Utd",
    "Tottenham Hotspur FC" : "Tottenham",
    "West Ham United FC" : "West Ham",
    "Everton FC" : "Everton",
    "Wolverhampton Wanderers FC" : "Wolves",
    "Leicester City FC" : "Leicester",
    "Ipswich Town FC" : "Ipswich",
    "Southamptom FC" : "Southamptom"
}



df['Team'] = df['Team'].replace(team_name_mapping)



print(df)  # Prints DataFrame in the console


save_path = r"C:\Users\d_par\OneDrive\Desktop\Danny\2025\Data Science\Portolio Projects\prem-xg-analysis\Data\current_epl_standings.csv"

df.to_csv(save_path, index=False)








