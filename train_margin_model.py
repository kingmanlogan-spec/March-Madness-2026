import difflib
import re

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


regular_season_games = pd.read_csv("MRegularSeasonCompactResults.csv")
tournament_games = pd.read_csv("MNCAATourneyCompactResults.csv")
teams = pd.read_csv("teams.csv")
team_lookup = pd.read_csv("MTeams.csv")

for column in [
    "adj_off",
    "adj_def",
    "net_rating",
    "sos_net",
    "eFG",
    "opp_eFG",
    "to_rate",
    "forced_to_rate",
    "orb_rate",
    "drb_rate",
    "ft_rate",
    "seed",
]:
    if column not in teams.columns:
        teams[column] = 0
    teams[column] = pd.to_numeric(teams[column], errors="coerce").fillna(0)

teams = teams.set_index("team")
team_id_to_name = dict(zip(team_lookup["TeamID"], team_lookup["TeamName"]))


def normalize_name(name: str) -> str:
    value = str(name).strip().lower()
    replacements = {
        "&": "and",
        " st.": " state",
        " st ": " state ",
        " st$": " state",
        " mt ": " mount ",
        " chr": " christian",
        " carolina st": " carolina state",
        " f dickinson": " fairleigh dickinson",
        " nc ": " north carolina ",
        " n carolina": " north carolina",
        " s carolina": " south carolina",
    }
    for old, new in replacements.items():
        value = re.sub(old, new, value)
    value = re.sub(r"[^a-z0-9 ]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


team_names = list(teams.index)
normalized_team_map = {normalize_name(name): name for name in team_names}


def resolve_team_name(name: str):
    normalized = normalize_name(name)
    if normalized in normalized_team_map:
        return normalized_team_map[normalized]
    matches = difflib.get_close_matches(normalized, normalized_team_map.keys(), n=1, cutoff=0.86)
    if matches:
        return normalized_team_map[matches[0]]
    return None


def build_features(t1, t2):
    return [
        t1.net_rating - t2.net_rating,
        t1.adj_off - t2.adj_off,
        t2.adj_def - t1.adj_def,
        t1.sos_net - t2.sos_net,
        t1.eFG - t2.opp_eFG,
        t1.to_rate - t2.forced_to_rate,
        t1.orb_rate - t2.drb_rate,
        t2.seed - t1.seed,
        t1.adj_tempo - t2.adj_tempo,
        (t1.adj_tempo + t2.adj_tempo) / 2,
        t1.ft_rate - t2.opp_ft_rate,
    ]


def clipped_margin(value: float) -> float:
    return max(-25.0, min(25.0, value))


games = pd.concat([regular_season_games, tournament_games], ignore_index=True)

X = []
y = []
seasons = []
skipped = 0

for _, g in games.iterrows():
    winner_name = team_id_to_name.get(g.WTeamID)
    loser_name = team_id_to_name.get(g.LTeamID)
    if winner_name is None or loser_name is None:
        skipped += 2
        continue

    team1_name = resolve_team_name(winner_name)
    team2_name = resolve_team_name(loser_name)
    if team1_name is None or team2_name is None:
        skipped += 2
        continue

    t1 = teams.loc[team1_name]
    t2 = teams.loc[team2_name]
    X.append(build_features(t1, t2))
    y.append(clipped_margin(float(g.WScore - g.LScore)))
    seasons.append(int(g.Season))

    X.append(build_features(t2, t1))
    y.append(clipped_margin(float(g.LScore - g.WScore)))
    seasons.append(int(g.Season))

rows = pd.DataFrame({
    "features": X,
    "target": y,
    "season": seasons,
})

season_list = sorted(rows["season"].unique())
validation_seasons = set(season_list[-3:])

train_rows = rows[~rows["season"].isin(validation_seasons)]
val_rows = rows[rows["season"].isin(validation_seasons)]

X_train = train_rows["features"].tolist()
y_train = train_rows["target"].tolist()
X_val = val_rows["features"].tolist()
y_val = val_rows["target"].tolist()

model = HistGradientBoostingRegressor(
    max_depth=4,
    learning_rate=0.05,
    max_iter=200,
    random_state=42,
)

model.fit(X_train, y_train)
preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
joblib.dump(model, "margin_model.pkl")

print("Margin model trained and saved")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Validation seasons:", sorted(validation_seasons))
print("Skipped rows:", skipped)
print("Validation MAE:", round(mae, 4))
