📊 EPL Expected Goals (xG) Analysis — 2023/24


-----------------------------------------------------------------------------------------------------------------------------


Overview

This project investigates the reliability and analytic value of Expected Goals (xG) — a widely used but often misunderstood metric in modern football. While xG is frequently cited by analysts, pundits, and clubs, most fans and even professionals struggle to explain how it's actually calculated or whether it deserves the level of trust it receives.

To explore this disconnect, I first conducted a short survey of 100 football fans:
✅ Most respondents believe xG is a useful stat
❌ But many don’t fully understand how it’s calculated




Interestingly, despite a general lack of understanding of the metric, football fans consider xG a significant insight into painting the picture of the match. 

These results led me to ask: Can we trust xG? And if so, what can it really tell us?


Thus, a full analysis of the 2023/24 Premier League season was conducted — exploring:

How accurate xG is as a predictive tool for goals
Which teams over- or under-perform their xG
Whether league standings would change if based only on xG
How team tactics and other factors shape xG trends

This analysis was built and documented through a series of Jupyter notebooks, which guide the entire journey from initial data collection and cleaning, to survey analysis, exploratory data visualizations, and final insights. The full workflow — including code, reasoning, and commentary — can be found in the /Notebooks folder.

As a final step, I built a Dash web app to bring the key visualizations and findings together in an interactive dashboard format. While the real depth lies in the notebooks, the app offers a polished, accessible summary for exploring xG trends across the 23/24 EPL season.

-----------------------------------------------------------------------------------------------------------------------------
🚀 Live Demo
🔗 You Can View the Live Dashboard at: https://epl23-24-xg-analysis-app.onrender.com


-----------------------------------------------------------------------------------------------------------------------------









🔍 Key Results & Applications

 The disparity, or ‘error’, between xG and actual goals scored in a game is normally distributed around 0 — suggesting that while xG is not perfect at predicting individual match outcomes, it is statistically unbiased and reliable over a large sample of games. 

xG Tables were found to have predictive power for future team form. This suggests xG could be used not only for retrospective analysis, but also as a tool for anticipating changes in teams momentum.
 
The interactive dashboard offers deep insight into league-wide and team-specific xG patterns — potentially useful for tactical preparation.

-----------------------------------------------------------------------------------------------------------------------------


📌 Future Work
Compare year-to-year xG league tables and actual tables to assess form prediction potential
Add player-specific xG dashboards and finishing efficiency ratings
Integrate non-shot xG opportunities (e.g., dangerous build up, pre-assist data) into a new or refined model to build on current statistics and paint an even more in depth picture of the game 


-----------------------------------------------------------------------------------------------------------------------------



 
📊 Screenshots
 See below still images of the dashboard menu and one of its tabs.








-----------------------------------------------------------------------------------------------------------------------------
📁 Folder Structure

EPL-xG-Analysis/
├── Notebooks/ # Jupyter notebooks for workflow including data cleaning, survey, analysis.
├── Data/ # Raw and cleaned CSV files
├── Additional\_Scripts/ # Dash\_App.py and App\_Functions.py
├── README.md
├── requirements.txt
├── Procfile # For Render deployment
├── LICENSE 
├── .gitignore 

----------------------

🧪 How to Run This Project Locally

You can interact with the full dashboard live here — no setup needed.

If you'd like to explore the more thorough, in depth analysis and data workflow behind the dashboard, follow the steps below to run the Jupyter notebooks locally.





1. Clone the Repository

bash 

git clone https://github.com/DannyPartington/EPL-xG-Analysis-.git
cd EPL-xG-Analysis-

-----------------------------------------------------------------------------------------------------------------------------

2. Install Dependencies
Make sure you have Python 3.10+ installed. Then install required packages:

Bash

pip install -r requirements.txt 

-----------------------------------------------------------------------------------------------------------------------------

3. Launch Jupyter Notebook
Bash

jupyter notebook 


Then open the .ipynb files inside the Notebooks/ folder and run through each cell step by step to:
Load and clean the data
Perform xG analysis
Generate and review the visualisations used in the dashboard
 

-----------------------------------------------------------------------------------------------------------------------------

📚 Tools & Libraries Used
 Pandas, Numpy, Plotly, Dash, Sklearn, HTML 

-----------------------------------------------------------------------------------------------------------------------------

📘 Data Sources
Public xG datasets scraped or aggregated from free football analytics resources
Team stats derived from match-level event data



