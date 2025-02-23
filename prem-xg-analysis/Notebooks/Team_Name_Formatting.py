# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:51:56 2025

@author: d_par
"""

# team_name_utils.py

import re

def standardize_team_name(team_name):
    """
    Standardizes team names from various formats to a consistent format.
    """
    # Dictionary mapping regex patterns to standardized names
    team_mapping = {
        r"manchester\s*city|manc\s*city|man\s*city": "Manchester City",
        r"manchester\s*united|man\s*united|man\s*utd|manc\s*united": "Manchester Utd",
        r"arsenal": "Arsenal",
        r"chelsea": "Chelsea",
        r"liverpool": "Liverpool",
        r"tottenham|spurs|tottenham\s*hotspur": "Tottenham",
        r"newcastle|newcastle\s*united": "Newcastle Utd",
        r"aston\s*villa": "Aston Villa",
        r"west\s*ham|west\s*ham\s*united": "West Ham",
        r"everton": "Everton",
        r"wolves|wolverhampton|wolverhampton\s*wanderers": "Wolves",
        r"brighton|brighton\s*&\s*hove|brighton\s*and\s*hove|bha": "Brighton",
        r"crystal\s*palace|palace": "Crystal Palace",
        r"fulham": "Fulham",
        r"brentford": "Brentford",
        r"nottingham\s*forest|nott\s*forest|notts\s*forest|forest": "Nott'ham Forest",
        r"bournemouth|afc\s*bournemouth": "Bournemouth",
        r"sheffield\s*united|sheff\s*united|sheffield\s*utd": "Sheffield Ud",
        r"luton\s*town|luton": "Luton Town",
        r"burnley": "Burnley"
    }
    
    # Normalize the input (strip whitespace and convert to lowercase)
    team_name = team_name.strip().lower()
    
    # Match against regex patterns
    for pattern, standardized_name in team_mapping.items():
        if re.search(pattern, team_name):  # Apply regex search
            return standardized_name
    
    # Return the original name if no match is found
    return team_name


def shorten_team_name(team_name):
    """
    Converts full team names into short identifiers.
    
    - "Manchester City" → "MC"
    - "Manchester Utd" → "MU"
    - All other teams → First three letters in uppercase
    
    Parameters:
    team_name (str): The full name of the football team.
    
    Returns:
    str: Shortened team name.
    """
    special_cases = {
        "Manchester City": "MC",
        "Manchester Utd": "MU"
    }
    
    return special_cases.get(team_name, team_name[:3].upper())

