import math

from pathlib import Path

import joblib

import pandas as pd


def load_teams(csv_path="teams.csv"):
    df = pd.read_csv(csv_path)
    df["team"] = df["team"].astype(str).str.strip()
    for column in [
        "adj_off",
        "adj_def",
        "adj_tempo",
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
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df


def get_team_list(csv_path="teams.csv"):
    df = load_teams(csv_path)
    return sorted(df["team"].dropna().unique().tolist())


def get_team_row(team_name: str, csv_path="teams.csv"):
    df = load_teams(csv_path)
    row = df[df["team"].str.lower() == team_name.strip().lower()]
    if row.empty:
        raise ValueError(f"Team not found: {team_name}")
    return row.iloc[0].to_dict()


def get_trained_model(model_path="model.pkl"):
    if not Path(model_path).exists():
        raise FileNotFoundError("model.pkl not found. Run train_model.py first.")
    return joblib.load(model_path)


def get_margin_model(model_path="margin_model.pkl"):
    if not Path(model_path).exists():
        return None
    return joblib.load(model_path)


def build_feature_vector(team1: dict, team2: dict):
    net_diff = float(team1.get("net_rating", 0)) - float(team2.get("net_rating", 0))
    off_diff = float(team1.get("adj_off", 0)) - float(team2.get("adj_off", 0))
    def_diff = float(team2.get("adj_def", 0)) - float(team1.get("adj_def", 0))
    sos_diff = float(team1.get("sos_net", 0)) - float(team2.get("sos_net", 0))
    efg_diff = float(team1.get("eFG", 0)) - float(team2.get("opp_eFG", 0))
    to_diff = float(team1.get("to_rate", 0)) - float(team2.get("forced_to_rate", 0))
    orb_diff = float(team1.get("orb_rate", 0)) - float(team2.get("drb_rate", 0))
    seed_diff = float(team2.get("seed", 8)) - float(team1.get("seed", 8))

    features = [[
        net_diff,
        off_diff,
        def_diff,
        sos_diff,
        efg_diff,
        to_diff,
        orb_diff,
        seed_diff,
    ]]

    signals = {
        "net_rating_diff": net_diff,
        "off_diff": off_diff,
        "def_diff": def_diff,
        "sos_diff": sos_diff,
        "efg_diff": efg_diff,
        "to_diff": to_diff,
        "orb_diff": orb_diff,
        "seed_diff": seed_diff,
    }

    return features, signals


def build_margin_feature_vector(team1: dict, team2: dict):
    base_features, _ = build_feature_vector(team1, team2)
    tempo_diff = float(team1.get("adj_tempo", 0)) - float(team2.get("adj_tempo", 0))
    adj_tempo_avg = (float(team1.get("adj_tempo", 0)) + float(team2.get("adj_tempo", 0))) / 2
    ft_rate_diff = float(team1.get("ft_rate", 0)) - float(team2.get("opp_ft_rate", 0))
    return [[
        *base_features[0],
        tempo_diff,
        adj_tempo_avg,
        ft_rate_diff,
    ]]


def predict_win_probability(team1_name: str, team2_name: str, csv_path="teams.csv"):
    team1 = get_team_row(team1_name, csv_path)
    team2 = get_team_row(team2_name, csv_path)
    model = get_trained_model()
    features, signals = build_feature_vector(team1, team2)

    team1_prob = model.predict_proba(features)[0][1]

    return {
        "team1": team1_name,
        "team2": team2_name,
        "team1_prob": team1_prob,
        "team2_prob": 1 - team1_prob,
        "predicted_winner": team1_name if team1_prob >= 0.5 else team2_name,
        "confidence": max(team1_prob, 1 - team1_prob),
        "features": features,
        "team1_row": team1,
        "team2_row": team2,
        "signals": signals,
    }


def heuristic_project_margin(signals):
    return (
        0.35 * signals["net_rating_diff"]
        + 0.15 * signals["off_diff"]
        + 0.15 * signals["def_diff"]
        + 0.10 * signals["sos_diff"]
    )


def project_margin(signals, features=None):
    if features is not None:
        margin_model = get_margin_model()
        if margin_model is not None:
            return float(margin_model.predict(features)[0])
    return heuristic_project_margin(signals)


def spread_to_market_prob(team1_spread: float) -> float:
    k = 0.13
    return 1 / (1 + math.exp(k * team1_spread))


def get_spread_edge(projected_margin: float, market_spread: float) -> float:
    return projected_margin + market_spread


def downgrade_recommendation(recommendation: str) -> str:
    order = ["PASS", "LEAN", "BET", "STRONG BET"]
    index = order.index(recommendation)
    return order[max(index - 1, 0)]


def get_recommendation(model_prob: float, market_prob: float, spread: float | None = None, spread_edge: float | None = None):
    prob_edge = model_prob - market_prob

    if spread_edge is not None:
        abs_spread_edge = abs(spread_edge)
        extreme_underdog_mismatch = spread is not None and spread > 12 and model_prob < 0.35

        if abs_spread_edge < 2.0:
            rec = "PASS"
        elif abs_spread_edge < 3.5:
            rec = "LEAN"
        elif abs_spread_edge < 5.0:
            rec = "BET"
        else:
            rec = "BET"

        if spread is not None and abs(spread) >= 12 and rec == "BET":
            rec = "BET"
        if extreme_underdog_mismatch and rec == "BET":
            rec = "BET"

        return rec, prob_edge

    if prob_edge < 0.03:
        rec = "PASS"
    elif prob_edge < 0.06:
        rec = "LEAN"
    elif prob_edge < 0.10:
        rec = "BET"
    else:
        rec = "STRONG BET"

    return rec, prob_edge


def analyze_matchup(team1_name: str, team2_name: str, spread: float | None = None, csv_path="teams.csv"):
    result = predict_win_probability(team1_name, team2_name, csv_path)
    margin_features = build_margin_feature_vector(result["team1_row"], result["team2_row"])
    projected_margin = project_margin(result["signals"], margin_features)
    analysis = {
        **result,
        "projected_margin": projected_margin,
        "bet_side": None,
    }

    if spread is not None:
        market_prob = spread_to_market_prob(spread)
        spread_edge = get_spread_edge(projected_margin, spread)
        spread_edge = max(min(spread_edge, 7), -7)
        recommendation, edge = get_recommendation(
            result["team1_prob"],
            market_prob,
            spread,
            spread_edge,
        )
        bet_side = None
        if spread_edge > 0:
            bet_side = f"{team1_name} ATS"
        elif spread_edge < 0:
            bet_side = f"{team2_name} ATS"
        analysis.update({
            "spread": spread,
            "market_prob": market_prob,
            "edge": edge,
            "recommendation": recommendation,
            "bet_side": bet_side,
            "model_spread": -projected_margin,
            "spread_edge": spread_edge,
        })

    return analysis
