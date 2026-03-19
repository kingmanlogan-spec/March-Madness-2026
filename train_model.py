import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import difflib
import re

games = pd.read_csv("games.csv")
teams = pd.read_csv("teams.csv")

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
    "seed",
]:
    if column not in teams.columns:
        teams[column] = 0
    teams[column] = pd.to_numeric(teams[column], errors="coerce").fillna(0)

teams = teams.set_index("team")


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
    ]

X = []
y = []
skipped = 0

for _, g in games.iterrows():
    team1_name = resolve_team_name(g.team1)
    team2_name = resolve_team_name(g.team2)

    if team1_name is None or team2_name is None:
        skipped += 1
        continue

    t1 = teams.loc[team1_name]
    t2 = teams.loc[team2_name]

    features = build_features(t1, t2)

    X.append(features)
    y.append(g.result)

model = LogisticRegression()

model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained and saved")
print("Training rows:", len(X))
print("Skipped rows:", skipped)
