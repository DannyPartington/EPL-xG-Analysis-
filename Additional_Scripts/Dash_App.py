import dash
import os
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from Additional_Scripts.App_Functions import (
    plot_xg_error_histogram, plot_xg_error_vs_total_goals,
    calculate_xg_league_table, calculate_xg_to_goals, 
    merge_and_rank_xg, style_xg_rankings,
    plot_home_vs_away_xg, plot_home_vs_away_xGA, plot_pl_vs_xg_ranking,
    generate_xg_summary_table,
    analyze_xg_vs_possession, analyze_xg_vs_formation,
    analyze_xg_vs_shots, analyze_xg_vs_month, analyze_xg_vs_time,
    generate_team_summary_table
)

# Initialize Dash app with Bootstrap (Dark Theme)
app = dash.Dash(__name__, title="EPL xG Dashboard", external_stylesheets=[dbc.themes.DARKLY])


# Get the absolute path to the directory containing Dash_App.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths for data files
file_path1 = os.path.join(BASE_DIR, '..', 'Data', 'team_data.csv')
file_path2 = os.path.join(BASE_DIR, '..', 'Data', 'cleaned_prem_data.csv')
file_path3 = os.path.join(BASE_DIR, '..', 'Data', 'avg_xg_table.csv')
file_path4 = os.path.join(BASE_DIR, '..', 'Data', 'avg_xga_table.csv')
file_path5 = os.path.join(BASE_DIR, '..', 'Data', 'Rankings.csv')
file_path6 = os.path.join(BASE_DIR, '..', 'Data', 'xg_league_table.csv')
file_path7 = os.path.join(BASE_DIR, '..', 'Data', 'xg_goal_ratio_table.csv')

# Load data
team_data = pd.read_csv(file_path1)
cleaned_prem_data = pd.read_csv(file_path2)
avg_xG = pd.read_csv(file_path3)
avg_xGA = pd.read_csv(file_path4)
league_table = pd.read_csv(file_path5)
xg_league_table = pd.read_csv(file_path6)
xg_goal_ratio_table = pd.read_csv(file_path7)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # ← tracks the page
    
    dbc.Container([
        dbc.Nav([
            dbc.NavItem(dcc.Link("Home", href="/", className="nav-link", style={
                'padding': '16px 24px',
                'fontSize': '18px',
                'fontWeight': 'bold',
                'color': 'white',
                'display': 'inline-block'
            })),
            dbc.NavItem(dcc.Link("xG Accuracy", href="/xg-error", className="nav-link", style={
                'padding': '16px 24px',
                'fontSize': '18px',
                'fontWeight': 'bold',
                'color': 'white',
                'display': 'inline-block'
            })),
            dbc.NavItem(dcc.Link("xG Creation & Conversion Analysis", href="/league-xg-analysis", className="nav-link", style={
                'padding': '16px 24px',
                'fontSize': '18px',
                'fontWeight': 'bold',
                'color': 'white',
                'display': 'inline-block'
            })),
            dbc.NavItem(dcc.Link("xG League Table", href="/xg-league-table", className="nav-link", style={
                'padding': '16px 24px',
                'fontSize': '18px',
                'fontWeight': 'bold',
                'color': 'white',
                'display': 'inline-block'
            })),
            dbc.NavItem(dcc.Link("Team xG Analysis", href="/team-xg-analysis", className="nav-link", style={
                'padding': '16px 24px',
                'fontSize': '18px',
                'fontWeight': 'bold',
                'color': 'white',
                'display': 'inline-block'
            })),
            dbc.NavItem(dcc.Link("About", href="/about", className="nav-link", style={
                'padding': '16px 24px',
                'fontSize': '18px',
                'fontWeight': 'bold',
                'color': 'white',
                'display': 'inline-block'
            })),
        ], pills=True, justified=True, style={
            "marginBottom": "20px",
            "backgroundColor": "#1e1e1e",
            "paddingTop": "10px",
            "paddingBottom": "10px"
        }),
    
        html.Div(id='page-content')
    ], fluid=True, style={'padding': '20px'})

])
   

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def render_content(pathname):
    
    if pathname == '/' or pathname == '/home':

        return dbc.Container([
            # Intro Section
            html.Div([
                html.H1("EPL Expected Goals (xG) Analysis — 2023/24", 
                        style={'textAlign': 'center', 'color': '#17a2b8', 'marginTop': '20px'}),
            
                html.P(
                    "This interactive dashboard provides an in-depth analysis of Expected Goals (xG) for the 2023/24 Premier League season. "
                    "Through data-driven insights, this tool examines the accuracy of xG as a predictive metric and explores "
                    "team and league-wide performance trends.",
                    style={'textAlign': 'center', 'color': 'white', 'marginBottom': '30px'}
                ),
            ], style={'maxWidth': '900px', 'margin': 'auto'}),
            html.Hr(style={'borderColor': '#17a2b8', 'borderWidth': '2px', 'margin': '40px 0'}),



           

            html.H2("Explore the Dashboard", style={'color': '#17a2b8', 'textAlign': 'center', 'marginTop': '20px'}),
            
            
            # Clickable Section Cards
           dbc.Row([
                dbc.Col(
                    dcc.Link(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("xG Error", className="card-title", style={'color': '#17a2b8', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '24px'}),
                                html.P("Assess the accuracy of xG predictions compared to actual goals.", className="card-text"),
                            ]),
                            style={'backgroundColor': '#1e1e1e', 'padding': '10px', 'cursor': 'pointer'}
                        ),
                        href="/xg-error",
                        refresh=False
                    ),
                    width=4
                ),
                dbc.Col(
                    dcc.Link(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("xG Creation & Conversion Analaysis", className="card-title", style={'color': '#17a2b8', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '24px'}),
                                html.P("Explore xG trends across the entire league, from chance creation to chance conversion.", className="card-text"),
                            ]),
                            style={'backgroundColor': '#1e1e1e', 'padding': '10px', 'cursor': 'pointer'}
                        ),
                        href="/league-xg-analysis",
                        refresh=False
                    ),
                    width=4
                ),
                dbc.Col(
                    dcc.Link(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("xG League Table", className="card-title", style={'color': '#17a2b8', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '24px'}),
                                html.P("What if games were decided by xG? See how the final league table would look and compare xG-based standings to real life for the 23/24 campaign.", className="card-text"),
                            ]),
                            style={'backgroundColor': '#1e1e1e', 'padding': '10px', 'cursor': 'pointer'}
                        ),
                        href="/xg-league-table",
                        refresh=False
                    ),
                    width=4
                )
            ], justify="center", style={'marginTop': '20px'}),



                                        
            dbc.Row([
                dbc.Col(
                    dcc.Link(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Team xG Analysis", className="card-title", style={'color': '#17a2b8', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '24px'}),
                                html.P("In depth, team specific xG analysis covering formation, shots, possession, and more.", className="card-text"),
                            ]),
                            style={'backgroundColor': '#1e1e1e', 'padding': '10px', 'cursor': 'pointer'}
                        ),
                        href="/team-xg-analysis",
                        refresh=False
                    ),
                    width=4
                ),
                dbc.Col(
                    dcc.Link(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("About", className="card-title", style={'color': '#17a2b8', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '24px'}),
                                html.P("Learn more about this project, the data used, and the methods applied.", className="card-text"),
                            ]),
                            style={'backgroundColor': '#1e1e1e', 'padding': '10px', 'cursor': 'pointer'}
                        ),
                        href="/about",
                        refresh=False
                    ),
                    width=4
                )
            ], justify="center", style={'marginTop': '20px'}),
            html.Hr(style={'borderColor': '#17a2b8', 'borderWidth': '2px', 'margin': '40px 0'}),



             # Why Are We Doing This? Section
             html.Div([
                 html.H2("Why Are We Doing This?", style={'color': '#17a2b8', 'textAlign': 'center', 'marginTop': '30px'}),
             
                 html.P(
                     "While still a relatively new metric, xG has become a staple in modern matchday analysis "
                     ,
                     style={'color': 'white', 'textAlign': 'center'}
                 ),
             
                 html.P(
                     "But just how useful is it?",
                     style={'color': 'white', 'textAlign': 'center'}
                 ),
             
                 html.P(
                     "What, if any insights can teams draw from the metric to gain an edge over opponents? "
                     "And could the introduction of xG open the floodgates of a new era for sophisticated statistical models in football analytics?",
                     style={'color': 'white', 'textAlign': 'center'}
                 ),
             
                 html.P(
                     "However, if xG and similar new era metrics are going to be used and analysed effectively, "
                     "it is crucial to first ensure their reliability.",
                     style={'color': 'white', 'textAlign': 'center'}
                 ),
             
                 html.P(
                     "Therefore this website both investigates the accuracy of xG, and presents a range of key insights around xG "
                     "from the 23/24 season.",
                     style={'color': 'white', 'textAlign': 'center', 'marginBottom': '30px'}
                 ),
             ], style={'maxWidth': '900px', 'margin': 'auto'}),
             
             html.Hr(style={'borderColor': '#17a2b8'}),


            # Footer / Small Credit Section
            html.Footer([
                html.P([
                    "Created as part of a data analytics investigation. ",
                    html.A("View on GitHub", href="https://github.com/DannyPartington", target="_blank", style={'color': '#17a2b8'})
                ], style={'textAlign': 'center', 'color': 'white', 'marginTop': '20px'})
            ])
        ], fluid=True)

            dbc.Container(
                html.Footer([
                    html.P([
                        "You can see more of my work and experience, or how to get in touch on my Website. ",
                        html.A("Here", 
                               href="https://dannypartington.github.io/Analytics-Portfolio/",
                               target="_blank",
                               style={'color': '#17a2b8'})
                    ])
                ], style={'textAlign': 'center', 'color': 'white', 'marginTop': '20px'}),
                fluid=True
            )


    elif pathname == '/about':

        return dbc.Container([
            html.H1("About This Project", style={'textAlign': 'center', 'color': '#17a2b8', 'marginTop': '20px'}),
    
            html.P(
                "This app is a data-driven investigation into Expected Goals (xG) in the Premier League for the 2023/24 season. "
                "It explores how xG compares to actual goals, analyzes team and league-wide xG trends, "
                "and evaluates the reliability of xG as a modern football metric.",
                style={'color': 'white', 'marginBottom': '30px'}
            ),
    
            html.H3("Key Insights & Recommendations", style={'color': '#17a2b8', 'marginTop': '30px'}),
            html.Ul([
                html.Li("There are many scenarios in football where observers would 'expect' a goal, even if a shot isn't taken.Incorporating non-shot expected goal scenarios would create a more comprehensive metric of the most threatening team."),
                html.Li("xG could be a predictive tool for turn of form, as xG standings more closley align with the following seasons standings rather than the corresponsing standings."),
                html.Li("Teams with low xG conversion rates could benefit from focusing on improving shot quality over volume."),
            ], style={'color': 'white'}),
    
            html.P([
                "To read the full analysis and detailed recommendations, explore my Jupyter Notebooks which can be found in my ",
                html.A("GitHub repository", href="https://github.com/DannyPartington/EPL-xG-Analysis-", target="_blank", style={'color': '#17a2b8'}),
                "."
            ], style={'color': 'white', 'marginTop': '15px'}),
            
            html.H3("Future Work", style={'color': '#17a2b8', 'marginTop': '30px'}),
            html.Ul([
                html.Li("Compare year-to-year xG League Tables vs actual standings to assess whether xG can be used as a predictive indicator of form shifts."),
                html.Li("Player-specific xG analysis, assigning ratings based on over- or under-performance.."),
                html.Li("Integrate multi-season trends for long term analysis and progression"),
            ], style={'color': 'white'}),
            
            html.H3("Application", style={'color': '#17a2b8', 'marginTop': '30px'}),
            html.Ul([
                html.Li("New, more complex mastchday metrics reliability must be ensured. This dashboard assesses the error margins for xG."),
                html.Li("Teams can use the tool for summary of own performances or to gain insight into opponents xG patterns."),
              
            ], style={'color': 'white'}),
    
            html.H3("Methodology", style={'color': '#17a2b8', 'marginTop': '30px'}),
            html.Ul([
                html.Li("Data cleaning and processing."),
                html.Li("Analysis of xG vs. actual goal discrepancies."),
                html.Li("Exploration of team-specific xG patterns."),
                html.Li("Comparison of league-wide xG trends and chance conversion rates."),
            ], style={'color': 'white'}),
    
            html.H3("Technologies Used", style={'color': '#17a2b8', 'marginTop': '30px'}),
            html.Ul([
                html.Li("Python (Pandas, NumPy) for data processing and analysis."),
                html.Li("Plotly & Dash for interactive data visualizations."),
                html.Li("Bootstrap for responsive styling."),
            ], style={'color': 'white'}),
    
            html.H3("Data Sources", style={'color': '#17a2b8', 'marginTop': '30px'}),
            html.P(
                "Data sourced from publicly available football analytics datasets.",
                style={'color': 'white'}
            ),
    
            html.Hr(style={'borderColor': '#17a2b8', 'marginTop': '30px'}),
    
            html.P([
                "More of my data analysis projects are available on my ",
                html.A("personal portfolio site", href="https://dannypartington.github.io/Analytics-Portfolio/", target="_blank", style={'color': '#17a2b8'}),
                "."
            ], style={'color': 'white', 'textAlign': 'center', 'marginTop': '20px'})
        ], fluid=True)

  
    elif pathname == '/xg-error':
        return dbc.Container([
            html.H1("xG Error Analysis", style={'textAlign': 'center', 'color': '#17a2b8'}),
            dcc.Graph(figure=plot_xg_error_histogram(cleaned_prem_data)),
            dcc.Graph(figure=plot_xg_error_vs_total_goals(cleaned_prem_data))
        ])
    elif pathname == '/xg-league-table':
        return dbc.Container([
            html.H1("xG League Table", style={'textAlign': 'center', 'color': '#17a2b8'}),
    
            #  Add explanatory text box
            html.Div(
                "Wins are defined by having 0.5 or more xG than the opponent. "
                "If two teams had a difference of less than 0.5 xG, the match results in a draw.",
                style={
                    'color': 'white',
                    'fontWeight': 'bold',
                    'fontSize': '20px',
                    'textAlign': 'center',
                    'margin': '20px auto',
                    'padding': '15px',
                    'backgroundColor': '#1e1e1e',
                    'border': '2px solid #17a2b8',
                    'borderRadius': '10px',
                    'maxWidth': '900px'
                }
            ),

        # Table
        dash_table.DataTable(
            data=calculate_xg_league_table(cleaned_prem_data, league_table),  
            columns=[{"name": col, "id": col} for col in ["Rank", "Team", "Points", "Scored", "Conceded", "Goal_Difference"]],
            style_table={'overflowX': 'auto', 'margin': '20px'},
            style_header={'backgroundColor': '#343a40', 'color': 'white', 'fontWeight': 'bold'},
            style_cell={'backgroundColor': '#1e1e1e', 'color': 'white', 'textAlign': 'center'},
            style_data_conditional=[
                {'if': {'row_index': 0}, 'borderBottom': '2px solid gold'},
                {'if': {'row_index': 3}, 'borderBottom': '2px solid cyan'},
                {'if': {'row_index': 4}, 'borderBottom': '2px solid orange'},
                {'if': {'row_index': 5}, 'borderBottom': '2px solid orange'},
                {'if': {'row_index': 16}, 'borderBottom': '2px solid red'}
            ]
        )
    ])

    elif pathname == '/league-xg-analysis':
        return dbc.Container([
            html.H1("League-Wide xG Analysis", style={'textAlign': 'center', 'color': '#17a2b8'}),
            dcc.RadioItems(
                id='league-xg-selection',
                options=[
                    {'label': 'Chance Creation', 'value': 'chance-creation'},
                    {'label': 'Chance Conversion', 'value': 'chance-conversion'}
                ],
                value='chance-creation',
                inline=True,
                labelStyle={'margin-right': '15px', 'color': 'white'}
            ),
            html.Div(id='league-xg-content')
        ])

                      
    elif pathname == '/team-xg-analysis':
        return dbc.Container([
            html.H1("Team xG Analysis", style={'textAlign': 'center', 'color': '#17a2b8'}),
            dbc.Row([
                dbc.Col(html.Label("Select a Team:", style={'fontSize': '20px', 'color': 'white'}), width=2),
                dbc.Col(dcc.Dropdown(
                    id='team-selector',
                    options=[{'label': team, 'value': team} for team in sorted(team_data['Home'].unique())],
                    value=sorted(team_data['Home'].unique())[0],  
                    clearable=False,
                    style={'color': 'black'}
                ), width=4)
            ], justify='center', style={'marginBottom': '20px'}),
    
            html.Div(id='team-summary-table'),  
            html.Br(),
    
            dbc.Row([
                dbc.Col(dcc.Graph(id='xg-vs-possession-home', style={'height': '350px'}), width=6),
                dbc.Col(dcc.Graph(id='xg-vs-possession-away', style={'height': '350px'}), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='xg-vs-formation', style={'height': '350px'}), width=6),
                dbc.Col(dcc.Graph(id='xg-vs-shots', style={'height': '350px'}), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='xg-vs-month', style={'height': '350px'}), width=6),
                dbc.Col(dcc.Graph(id='xg-vs-time', style={'height': '350px'}), width=6),
            ]),
        ])



    


@app.callback(
    Output('team-summary-table', 'children'),  
    Output('xg-vs-possession-home', 'figure'),
    Output('xg-vs-possession-away', 'figure'),
    Output('xg-vs-formation', 'figure'),
    Output('xg-vs-shots', 'figure'),
    Output('xg-vs-month', 'figure'),
    Output('xg-vs-time', 'figure'),
    Input('team-selector', 'value')
)
def update_team_analysis(team):
    #Generate summary table for the selected team
    summary_table = generate_team_summary_table(
        team, xg_goal_ratio_table, league_table, xg_league_table, avg_xG, avg_xGA
    )

    #Generate all the graphs
    fig_home, fig_away = analyze_xg_vs_possession(team, team_data)
    formation_fig = analyze_xg_vs_formation(team, team_data)
    shots_fig = analyze_xg_vs_shots(team, team_data)
    month_fig = analyze_xg_vs_month(team, team_data)
    time_fig = analyze_xg_vs_time(team, team_data)

    return summary_table, fig_home, fig_away, formation_fig, shots_fig, month_fig, time_fig


@app.callback(
    Output('league-xg-content', 'children'), 
    Input('league-xg-selection', 'value')
)
def update_league_xg_content(selected_section):
    print(f"Tab selected: {selected_section}")  

    if selected_section == 'chance-creation':
        return dbc.Container([
            dbc.Card(
                dcc.Graph(figure=plot_home_vs_away_xg(avg_xG), style={'width': '100%', 'height': '500px'}),
                style={'border': '2px solid #444', 'border-radius': '10px', 'padding': '10px', 'margin-bottom': '20px'}
            ),
            dbc.Card(
                dcc.Graph(figure=plot_home_vs_away_xGA(avg_xGA), style={'width': '100%', 'height': '500px'}),
                style={'border': '2px solid #444', 'border-radius': '10px', 'padding': '10px'})
        ])

    elif selected_section == 'chance-conversion':
        data = calculate_xg_to_goals(cleaned_prem_data, league_table)

        if data is None or len(data) == 0:
            return html.Div("Error: No data returned from function.", style={'color': 'red'})

        return dbc.Row([
            dbc.Col([
                html.H5("xG Conversion Rank", style={'textAlign': 'center', 'color': '#17a2b8', 'margin-bottom': '10px'}),
            
                dash_table.DataTable(
                    data=[
                        {"Rank": i + 1, **row}
                        for i, row in enumerate(data.to_dict('records'))
                    ],
                    columns=[
                        {"name": "Rank", "id": "Rank"},
                        {"name": "Team", "id": "Team"},
                        {"name": "xG per Goal", "id": "xG_to_Goals_Ratio"}
                    ],
                    style_table={'overflowX': 'auto', 'margin': '10px'},
                    style_header={'backgroundColor': '#343a40', 'color': 'white', 'fontWeight': 'bold'},
                    style_cell={'backgroundColor': '#1e1e1e', 'color': 'white', 'textAlign': 'center'},
                    sort_action="native"
                )
            ], width=4, style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center'}),
             # Table Column
        
            dbc.Col([
               html.H5("Rank Shift Chart", style={
                'textAlign': 'center',
                'color': '#17a2b8',
                'marginTop': '60px',
                'fontWeight': 'bold',
                'fontSize': '22px'
            }),
            html.P(
                "The below graph comparing final league position with xG conversion rank gives an insight into the impact of missing or taking the chances you're 'expected' to score.",
                style={
                    'textAlign': 'center',
                    'color': 'white',
                    'fontSize': '16px',
                    'marginBottom': '10px'
                }
            ),
            dcc.Graph(figure=plot_pl_vs_xg_ranking(data), style={'width': '100%', 'height': '700px', 'marginTop': '10px'})
            ])
            
        ])


server = app.server

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))











