from pathlib import Path

import pandas as pd

teams = pd.read_csv("MTeams.csv")
games_file = "RegularSeasonCompactResults.csv"
if not Path(games_file).exists():
    games_file = "MRegularSeasonCompactResults.csv"
games = pd.read_csv(games_file)

team_map = dict(zip(teams.TeamID, teams.TeamName))

rows = []

for _, g in games.iterrows():

    team1 = team_map[g.WTeamID]
    team2 = team_map[g.LTeamID]

    rows.append({
        "team1": team1,
        "team2": team2,
        "result": 1
    })

    rows.append({
        "team1": team2,
        "team2": team1,
        "result": 0
    })

df = pd.DataFrame(rows)

df.to_csv("games.csv", index=False)

print("Saved games.csv")
print(df.head())
