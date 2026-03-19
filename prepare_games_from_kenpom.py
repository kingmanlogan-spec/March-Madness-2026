from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
TARGET_SEASON = 2026


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE / name)


def build_name_mapping() -> dict:
    """Map KenPom team names to the scanner/app's ESPN-style team names when available."""
    mapping = load_csv("REF _ NCAAM Conference and ESPN Team Name Mapping.csv")
    name_map = dict(zip(mapping["TeamName"].astype(str).str.strip(), mapping["Mapped ESPN Team Name"].astype(str).str.strip()))

    # Fix known typo in uploaded postseason file.
    name_map["LIU"] = "Long Island Unviersity"
    return name_map


def main() -> None:
    name_map = build_name_mapping()

    summary = load_csv("INT _ KenPom _ Summary.csv")
    offense = load_csv("INT _ KenPom _ Offense.csv")
    defense = load_csv("INT _ KenPom _ Defense.csv")
    misc = load_csv("INT _ KenPom _ Miscellaneous Team Stats.csv")
    tournament = load_csv("REF _ Post-Season Tournament Teams.csv")

    summary = summary.loc[summary["Season"] == TARGET_SEASON, [
        "Season", "TeamName", "AdjOE", "AdjDE", "AdjTempo", "AdjEM"
    ]].copy()

    offense = offense.loc[offense["Season"] == TARGET_SEASON, [
        "Season", "TeamName", "eFGPct", "TOPct", "ORPct", "FTRate"
    ]].copy()

    defense = defense.loc[defense["Season"] == TARGET_SEASON, [
        "Season", "TeamName", "eFGPct", "TOPct", "ORPct", "FTRate"
    ]].copy().rename(columns={
        "eFGPct": "opp_eFG",
        "TOPct": "forced_to_rate",
        "ORPct": "opp_orb_rate",
        "FTRate": "opp_ft_rate",
    })

    misc = misc.loc[misc["Season"] == TARGET_SEASON, [
        "Season", "TeamName", "FG3Rate", "OppFG3Rate", "StlRate", "OppStlRate"
    ]].copy().rename(columns={
        "FG3Rate": "fg3_rate",
        "OppFG3Rate": "opp_fg3_rate",
        "StlRate": "stl_rate",
        "OppStlRate": "opp_stl_rate",
    })

    tournament = tournament.loc[
        (tournament["Season"] == TARGET_SEASON)
        & (tournament["Post-Season Tournament"] == "March Madness"),
        ["Season", "Team Name", "Seed", "Region"]
    ].copy().rename(columns={"Team Name": "team"})

    for df in [summary, offense, defense, misc]:
        df["team"] = df["TeamName"].astype(str).str.strip().map(lambda x: name_map.get(x, x))

    teams = (
        summary.rename(columns={
            "AdjOE": "adj_off",
            "AdjDE": "adj_def",
            "AdjTempo": "adj_tempo",
            "AdjEM": "net_rating",
        })
        .drop(columns=["TeamName"])
        .merge(
            offense.drop(columns=["TeamName", "Season"]).rename(columns={
                "eFGPct": "eFG",
                "TOPct": "to_rate",
                "ORPct": "orb_rate",
                "FTRate": "ft_rate",
            }),
            on="team",
            how="left",
        )
        .merge(
            defense.drop(columns=["TeamName", "Season"]),
            on="team",
            how="left",
        )
        .merge(
            misc.drop(columns=["TeamName", "Season"]),
            on="team",
            how="left",
        )
        .merge(tournament[["team", "Seed", "Region"]], on="team", how="left")
    )

    # Defensive rebounding rate is the complement of opponent offensive rebounding rate.
    teams["drb_rate"] = 100 - teams["opp_orb_rate"]

    # Keep seed numeric for the model.
    teams = teams.rename(columns={"Seed": "seed", "Region": "region"})
    teams["seed"] = pd.to_numeric(teams["seed"], errors="coerce")

    # Order columns for the model/scanner.
    ordered_cols = [
        "team",
        "seed",
        "region",
        "adj_off",
        "adj_def",
        "adj_tempo",
        "net_rating",
        "eFG",
        "opp_eFG",
        "to_rate",
        "forced_to_rate",
        "orb_rate",
        "drb_rate",
        "ft_rate",
        "opp_ft_rate",
        "fg3_rate",
        "opp_fg3_rate",
        "stl_rate",
        "opp_stl_rate",
    ]
    teams = teams[ordered_cols].sort_values(["seed", "team"], na_position="last").reset_index(drop=True)

    # Save the enriched model input file.
    output_path = BASE / "teams.csv"
    teams.to_csv(output_path, index=False)

    # Also save a diagnostics file so it's easy to inspect coverage.
    diagnostics = pd.DataFrame([
        {"metric": "rows", "value": len(teams)},
        {"metric": "teams_with_seed", "value": int(teams["seed"].notna().sum())},
        {"metric": "teams_missing_any_core_feature", "value": int(teams[["adj_off", "adj_def", "net_rating", "eFG", "opp_eFG", "to_rate", "forced_to_rate", "orb_rate", "drb_rate"]].isna().any(axis=1).sum())},
    ])
    diagnostics.to_csv(BASE / "teams_build_diagnostics.csv", index=False)

    print(f"Saved {output_path.name} with {len(teams)} teams for season {TARGET_SEASON}.")
    print(f"Tournament seeds attached for {int(teams['seed'].notna().sum())} teams.")
    print(teams.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
