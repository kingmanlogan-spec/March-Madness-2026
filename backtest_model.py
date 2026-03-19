from __future__ import annotations

import argparse
import math
import unicodedata
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from model import build_feature_vector, build_margin_feature_vector, get_trained_model, get_recommendation, get_spread_edge, project_margin, spread_to_market_prob

BASE = Path(__file__).resolve().parent


MANUAL_ALIASES = {
    # Kaggle -> KenPom / common names
    "abilene chr": "abilene christian",
    "alabama st": "alabama state",
    "suny albany": "ualbany",
    "alcorn st": "alcorn state",
    "american univ": "american",
    "appalachian st": "app state",
    "arizona st": "arizona state",
    "ark little rock": "little rock",
    "ark pine bluff": "arkansas pine bluff",
    "arkansas st": "arkansas state",
    "ball st": "ball state",
    "boise st": "boise state",
    "boston univ": "boston university",
    "c michigan": "central michigan",
    "cent arkansas": "central arkansas",
    "central conn": "central connecticut",
    "charleston so": "charleston southern",
    "coastal car": "coastal carolina",
    "col charleston": "charleston",
    "colorado st": "colorado state",
    "delaware st": "delaware state",
    "e illinois": "eastern illinois",
    "e kentucky": "eastern kentucky",
    "e michigan": "eastern michigan",
    "e washington": "eastern washington",
    "etsu": "east tennessee state",
    "f dickinson": "fairleigh dickinson",
    "fl atlantic": "florida atlantic",
    "fgcu": "florida gulf coast",
    "florida intl": "florida international",
    "florida st": "florida state",
    "g washington": "george washington",
    "ga southern": "georgia southern",
    "georgia st": "georgia state",
    "idaho st": "idaho state",
    "il chicago": "illinois chicago",
    "illinois st": "illinois state",
    "indiana st": "indiana state",
    "iowa st": "iowa state",
    "pfw": "purdue fort wayne",
    "iupui": "iu indianapolis",
    "jackson st": "jackson state",
    "jacksonville st": "jacksonville state",
    "kansas st": "kansas state",
    "kent": "kent state",
    "long beach st": "long beach state",
    "liu brooklyn": "liu",
    "liu": "long island university",
    "loy marymount": "loyola marymount",
    "ma lowell": "umass lowell",
    "mcneese st": "mcneese",
    "md e shore": "maryland eastern shore",
    "michigan st": "michigan state",
    "miami fl": "miami",
    "mississippi st": "mississippi state",
    "mississippi": "ole miss",
    "missouri kc": "umkc",
    "missouri st": "missouri state",
    "monmouth nj": "monmouth",
    "montana st": "montana state",
    "morehead st": "morehead state",
    "murray st": "murray state",
    "nc a&t": "north carolina a&t",
    "nc central": "north carolina central",
    "nc state": "nc state",
    "neb omaha": "omaha",
    "nicholls st": "nicholls",
    "n colorado": "northern colorado",
    "n dakota st": "north dakota state",
    "n kentucky": "northern kentucky",
    "n florida": "north florida",
    "new mexico st": "new mexico state",
    "norfolk st": "norfolk state",
    "north carolina st": "nc state",
    "northwestern la": "northwestern state",
    "ohio st": "ohio state",
    "oklahoma st": "oklahoma state",
    "oregon st": "oregon state",
    "penn st": "penn state",
    "portland st": "portland state",
    "s carolina st": "south carolina state",
    "cs fullerton": "cal state fullerton",
    "s dakota st": "south dakota state",
    "s illinois": "southern illinois",
    "s methodist": "smu",
    "s mississippi": "southern miss",
    "s utah": "southern utah",
    "sacred heart": "sacred heart",
    "san diego st": "san diego state",
    "se louisiana": "southeastern louisiana",
    "se missouri st": "southeast missouri state",
    "sf austin": "stephen f austin",
    "southern utah": "southern utah",
    "st bonaventure": "st bonaventure",
    "st francis ny": "st francis brooklyn",
    "st francis pa": "saint francis",
    "st john's": "st john's",
    "st joseph s pa": "st joseph s",
    "st joseph's pa": "st joseph s",
    "st louis": "saint louis",
    "st mary s ca": "st mary s",
    "st marys ca": "st mary s",
    "st mary's ca": "st mary s",
    "st mary's": "st mary s",
    "st peter's": "saint peter's",
    "stanford": "stanford",
    "stephen f austin": "stephen f austin",
    "tcu": "tcu",
    "tennessee st": "tennessee state",
    "texas a&m": "texas a&m",
    "texas a&m cc": "texas a&m corpus chris",
    "texas st": "texas state",
    "tx southern": "texas southern",
    "the citadel": "citadel",
    "uab": "uab",
    "uc davis": "uc davis",
    "uc irvine": "uc irvine",
    "uc riverside": "uc riverside",
    "uc santa barbara": "uc santa barbara",
    "ucf": "ucf",
    "ucla": "ucla",
    "uconn": "connecticut",
    "uiw": "incarnate word",
    "umass": "massachusetts",
    "umass lowell": "umass lowell",
    "umbc": "umbc",
    "unc asheville": "unc asheville",
    "unc greensboro": "unc greensboro",
    "unc wilmington": "unc wilmington",
    "unlv": "unlv",
    "usc": "usc",
    "suny albany": "ualbany",
    "ut arlington": "ut arlington",
    "ut martin": "ut martin",
    "ut rio grande valley": "ut rio grande valley",
    "ut san antonio": "utsa",
    "utah st": "utah state",
    "vcu": "vcu",
    "virginia st": "virginia state",
    "washington st": "washington state",
    "weber st": "weber state",
    "cleveland st": "cleveland state",
    "penn": "pennsylvania",
    "mtsu": "middle tennessee",
    "tam c christi": "texas a and m corpus chris",
    "tam c. christi": "texas a and m corpus chris",
    "wi milwaukee": "milwaukee",
    "wku": "western kentucky",
    "west virginia": "west virginia",
    "wichita st": "wichita state",
    "wright st": "wright state",
    "youngstown st": "youngstown state",
}


REQUIRED_FEATURE_COLUMNS = [
    "adj_off",
    "adj_def",
    "net_rating",
    "eFG",
    "opp_eFG",
    "to_rate",
    "forced_to_rate",
    "orb_rate",
    "drb_rate",
    "seed",
]


def normalize_name(name: str) -> str:
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("&", " and ")
    s = s.replace("mt.", "mount ").replace("mt ", "mount ")
    s = s.replace("st.", "st ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\bsaint\b", "st", s)
    s = re.sub(r"\buniv\b|\buniversity\b", "", s)
    s = re.sub(r"\bcollege\b", "", s)
    s = re.sub(r"\bthe\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return MANUAL_ALIASES.get(s, s)


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE / name)


def build_name_mapping() -> dict[str, str]:
    mapping = load_csv("REF _ NCAAM Conference and ESPN Team Name Mapping.csv")
    out: dict[str, str] = {}
    for _, row in mapping.iterrows():
        src = normalize_name(row["TeamName"])
        dst = str(row["Mapped ESPN Team Name"]).strip()
        out[src] = dst
        out.setdefault(normalize_name(dst), dst)
    return out


def build_historical_teams() -> pd.DataFrame:
    name_map = build_name_mapping()

    summary = load_csv("INT _ KenPom _ Summary.csv")[
        ["Season", "TeamName", "AdjOE", "AdjDE", "AdjTempo", "AdjEM"]
    ].copy()
    offense = load_csv("INT _ KenPom _ Offense.csv")[
        ["Season", "TeamName", "eFGPct", "TOPct", "ORPct", "FTRate"]
    ].copy()
    defense = load_csv("INT _ KenPom _ Defense.csv")[
        ["Season", "TeamName", "eFGPct", "TOPct", "ORPct", "FTRate"]
    ].copy().rename(
        columns={
            "eFGPct": "opp_eFG",
            "TOPct": "forced_to_rate",
            "ORPct": "opp_orb_rate",
            "FTRate": "opp_ft_rate",
        }
    )
    seeds = load_csv("MNCAATourneySeeds.csv").copy()
    seeds["seed"] = seeds["Seed"].astype(str).str.extract(r"(\d+)").astype(float)
    teams_lookup = load_csv("MTeams.csv")[['TeamID', 'TeamName']].copy()
    seeds = seeds.merge(teams_lookup, on="TeamID", how="left")[["Season", "TeamName", "seed"]]

    for df in [summary, offense, defense]:
        df["team"] = df["TeamName"].astype(str).str.strip().map(
            lambda x: name_map.get(normalize_name(x), str(x).strip())
        )
        df["team_norm"] = df["team"].map(normalize_name)

    seeds["team"] = seeds["TeamName"].astype(str).str.strip()
    seeds["team_norm"] = seeds["team"].map(normalize_name)

    teams = (
        summary.rename(
            columns={
                "AdjOE": "adj_off",
                "AdjDE": "adj_def",
                "AdjTempo": "adj_tempo",
                "AdjEM": "net_rating",
            }
        )[["Season", "team", "team_norm", "adj_off", "adj_def", "adj_tempo", "net_rating"]]
        .merge(
            offense[["Season", "team_norm", "eFGPct", "TOPct", "ORPct", "FTRate"]].rename(
                columns={"eFGPct": "eFG", "TOPct": "to_rate", "ORPct": "orb_rate", "FTRate": "ft_rate"}
            ),
            on=["Season", "team_norm"],
            how="left",
        )
        .merge(
            defense[["Season", "team_norm", "opp_eFG", "forced_to_rate", "opp_orb_rate", "opp_ft_rate"]],
            on=["Season", "team_norm"],
            how="left",
        )
        .merge(
            seeds[["Season", "team_norm", "seed"]],
            on=["Season", "team_norm"],
            how="left",
        )
    )
    teams["drb_rate"] = 100 - teams["opp_orb_rate"]
    teams["sos_net"] = 0.0
    teams = teams.drop_duplicates(subset=["Season", "team_norm"]).reset_index(drop=True)
    return teams


def build_season_lookups(teams: pd.DataFrame):
    season_to_exact = {}
    season_to_norm = {}
    for season, grp in teams.groupby("Season"):
        season_to_exact[season] = {str(t).strip(): rec for t, rec in zip(grp["team"], grp.to_dict(orient="records"))}
        season_to_norm[season] = {str(t).strip(): rec for t, rec in zip(grp["team_norm"], grp.to_dict(orient="records"))}
    return season_to_exact, season_to_norm


def resolve_team(raw_name: str, season: int, season_exact: dict, season_norm: dict):
    raw = str(raw_name).strip()
    if season not in season_exact:
        return None, "season_missing"
    if raw in season_exact[season]:
        return season_exact[season][raw], "exact"
    norm = normalize_name(raw)
    if norm in season_norm[season]:
        return season_norm[season][norm], "normalized"
    # soft fallback for one strong match only
    from difflib import get_close_matches
    matches = get_close_matches(norm, list(season_norm[season].keys()), n=2, cutoff=0.94)
    if len(matches) == 1:
        return season_norm[season][matches[0]], "fuzzy"
    return None, f"team_unmatched:{raw}"


def brier_score(y_true, y_prob):
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true) if y_true else float("nan")


def log_loss(y_true, y_prob, eps=1e-15):
    probs = [min(max(p, eps), 1 - eps) for p in y_prob]
    return -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for y, p in zip(y_true, probs)) / len(y_true) if y_true else float("nan")


def bucket_spread(spread: float) -> str:
    a = abs(float(spread))
    if a < 3:
        return "0-3"
    if a < 7:
        return "3-7"
    if a < 12:
        return "7-12"
    return "12+"


def load_spreads(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    spreads = pd.read_csv(path)
    req = {"season", "team1", "team2", "spread"}
    missing = req - set(map(str.lower, spreads.columns))
    spreads.columns = [c.lower() for c in spreads.columns]
    if missing:
        raise ValueError(f"Spreads file missing columns: {sorted(missing)}")
    spreads["team1_norm"] = spreads["team1"].map(normalize_name)
    spreads["team2_norm"] = spreads["team2"].map(normalize_name)
    return spreads


def main():
    parser = argparse.ArgumentParser(description="Backtest March Madness model with skip diagnostics.")
    parser.add_argument("--games-csv", default="MNCAATourneyDetailedResults.csv")
    parser.add_argument("--spreads-csv", default=None)
    parser.add_argument("--model-path", default="model.pkl")
    parser.add_argument("--results-out", default="backtest_results.csv")
    parser.add_argument("--skip-out", default="backtest_skip_summary.csv")
    args = parser.parse_args()

    teams = build_historical_teams()
    season_exact, season_norm = build_season_lookups(teams)
    model = get_trained_model(BASE / args.model_path)

    games = load_csv(args.games_csv)
    team_ids = load_csv("MTeams.csv")[["TeamID", "TeamName"]]
    id_to_name = dict(zip(team_ids["TeamID"], team_ids["TeamName"]))
    spreads = load_spreads(str(BASE / args.spreads_csv) if args.spreads_csv else None)
    spread_lookup = None
    if spreads is not None:
        spread_lookup = {
            (int(r.season), r.team1_norm, r.team2_norm): float(r.spread)
            for _, r in spreads.iterrows()
            if pd.notna(r.spread)
        }

    rows = []
    skip_counter = Counter()
    skip_by_season = defaultdict(Counter)
    resolve_counter = Counter()

    y_true = []
    y_prob = []
    outright_correct = 0
    margin_errors = []

    ats_total = ats_wins = ats_pushes = 0
    ats_by_rec = defaultdict(lambda: Counter())
    ats_by_bucket = defaultdict(lambda: Counter())

    for _, g in games.iterrows():
        season = int(g["Season"])
        winner_name = id_to_name.get(int(g["WTeamID"]))
        loser_name = id_to_name.get(int(g["LTeamID"]))
        if not winner_name or not loser_name:
            skip_counter["missing_teamid_name"] += 2
            skip_by_season[season]["missing_teamid_name"] += 2
            continue

        game_pairs = [
            (winner_name, loser_name, 1, int(g["WScore"]) - int(g["LScore"])),
            (loser_name, winner_name, 0, int(g["LScore"]) - int(g["WScore"])),
        ]

        for team1_name, team2_name, actual, actual_margin in game_pairs:
            team1, reason1 = resolve_team(team1_name, season, season_exact, season_norm)
            team2, reason2 = resolve_team(team2_name, season, season_exact, season_norm)
            resolve_counter[reason1] += 1
            resolve_counter[reason2] += 1

            if team1 is None:
                skip_counter[f"team1_{reason1}"] += 1
                skip_by_season[season][f"team1_{reason1}"] += 1
                continue
            if team2 is None:
                skip_counter[f"team2_{reason2}"] += 1
                skip_by_season[season][f"team2_{reason2}"] += 1
                continue

            if pd.isna(team1.get("seed")):
                team1["seed"] = 8
            if pd.isna(team2.get("seed")):
                team2["seed"] = 8

            missing_features = [c for c in REQUIRED_FEATURE_COLUMNS if pd.isna(team1.get(c)) or pd.isna(team2.get(c))]
            if missing_features:
                key = f"missing_features:{','.join(missing_features[:3])}{'...' if len(missing_features) > 3 else ''}"
                skip_counter[key] += 1
                skip_by_season[season][key] += 1
                continue

            features, signals = build_feature_vector(team1, team2)
            margin_features = build_margin_feature_vector(team1, team2)
            team1_prob = float(model.predict_proba(features)[0][1])
            predicted = 1 if team1_prob >= 0.5 else 0
            projected_margin = float(project_margin(signals, margin_features))

            outright_correct += int(predicted == actual)
            y_true.append(actual)
            y_prob.append(team1_prob)
            margin_errors.append(abs(projected_margin - actual_margin))

            row = {
                "season": season,
                "team1": team1_name,
                "team2": team2_name,
                "team1_resolved": team1["team"],
                "team2_resolved": team2["team"],
                "actual_result": actual,
                "team1_prob": team1_prob,
                "predicted_result": predicted,
                "predicted_winner": team1_name if predicted == 1 else team2_name,
                "actual_margin": actual_margin,
                "projected_margin": projected_margin,
                "margin_abs_error": abs(projected_margin - actual_margin),
            }

            if spread_lookup is not None:
                spread = spread_lookup.get((season, normalize_name(team1_name), normalize_name(team2_name)))
                if spread is None:
                    spread = spread_lookup.get((season, normalize_name(team1["team"]), normalize_name(team2["team"])))
                if spread is not None:
                    market_prob = spread_to_market_prob(spread)
                    recommendation, edge = get_recommendation(team1_prob, market_prob, spread)
                    spread_edge = get_spread_edge(projected_margin, spread)
                    ats_margin = actual_margin + spread
                    if ats_margin > 0:
                        ats_result = "win"
                        ats_wins += 1
                        ats_total += 1
                        ats_by_rec[recommendation]["win"] += 1
                        ats_by_bucket[bucket_spread(spread)]["win"] += 1
                    elif ats_margin < 0:
                        ats_result = "loss"
                        ats_total += 1
                        ats_by_rec[recommendation]["loss"] += 1
                        ats_by_bucket[bucket_spread(spread)]["loss"] += 1
                    else:
                        ats_result = "push"
                        ats_pushes += 1
                        ats_by_rec[recommendation]["push"] += 1
                        ats_by_bucket[bucket_spread(spread)]["push"] += 1

                    row.update({
                        "spread": spread,
                        "market_prob": market_prob,
                        "edge": edge,
                        "recommendation": recommendation,
                        "spread_edge": spread_edge,
                        "ats_result": ats_result,
                        "spread_bucket": bucket_spread(spread),
                    })
                else:
                    row.update({"spread": None})

            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(BASE / args.results_out, index=False)

    skip_rows = []
    for reason, count in skip_counter.most_common():
        skip_rows.append({"season": "ALL", "reason": reason, "count": count})
    for season, ctr in sorted(skip_by_season.items()):
        for reason, count in ctr.most_common():
            skip_rows.append({"season": season, "reason": reason, "count": count})
    for reason, count in resolve_counter.most_common():
        skip_rows.append({"season": "ALL", "reason": f"resolve:{reason}", "count": count})
    pd.DataFrame(skip_rows).to_csv(BASE / args.skip_out, index=False)

    evaluated = len(y_true)
    skipped = int(sum(skip_counter.values()))

    print(f"Games evaluated: {evaluated}")
    print(f"Games skipped: {skipped}")
    if evaluated:
        print(f"Outright accuracy: {outright_correct / evaluated:.3%}")
        print(f"Brier score: {brier_score(y_true, y_prob):.3f}")
        print(f"Log loss: {log_loss(y_true, y_prob):.3f}")
        print(f"Margin MAE: {sum(margin_errors) / len(margin_errors):.3f}")

    if skip_counter:
        print("\nTop skip reasons:")
        for reason, count in skip_counter.most_common(10):
            print(f"  {reason}: {count}")

    if ats_total:
        print(f"\nATS win rate: {ats_wins / ats_total:.3%} (pushes: {ats_pushes})")
        print("\nATS by recommendation tier:")
        for rec in ["PASS", "LEAN", "BET", "STRONG BET"]:
            ctr = ats_by_rec.get(rec)
            if not ctr:
                continue
            denom = ctr["win"] + ctr["loss"]
            if denom:
                print(f"  {rec}: {ctr['win'] / denom:.3%} ({ctr['win']}-{ctr['loss']}, pushes={ctr['push']})")
        print("\nATS by spread bucket:")
        for bucket in ["0-3", "3-7", "7-12", "12+"]:
            ctr = ats_by_bucket.get(bucket)
            if not ctr:
                continue
            denom = ctr["win"] + ctr["loss"]
            if denom:
                print(f"  {bucket}: {ctr['win'] / denom:.3%} ({ctr['win']}-{ctr['loss']}, pushes={ctr['push']})")

    print(f"\nSaved row-level results to {BASE / args.results_out}")
    print(f"Saved skip summary to {BASE / args.skip_out}")


if __name__ == "__main__":
    main()
