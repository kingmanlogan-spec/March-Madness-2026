import json
import ssl
import re
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from model import analyze_matchup, get_team_list


TEAM_NAME_MAP = {
    "Miami Ohio RedHawks": "Miami (OH)",
    "Southern Methodist Mustangs": "SMU",
    "Saint Mary's Gaels": "Saint Mary's",
    "Saint Marys Gaels": "Saint Mary's",
    "St. Mary's Gaels": "Saint Mary's",
    "UConn Huskies": "Connecticut",
    "UMass Minutemen": "Massachusetts",
    "Ole Miss Rebels": "Mississippi",
    "NC State Wolfpack": "NC State",
    "UNLV Rebels": "UNLV",
    "UIC Flames": "Illinois Chicago",
    "USC Trojans": "USC",
    "BYU Cougars": "BYU",
    "VCU Rams": "VCU",
    "UTSA Roadrunners": "UTSA",
    "UT Arlington Mavericks": "UT Arlington",
    "UT Martin Skyhawks": "UT Martin",
    "UTEP Miners": "UTEP",
    "Saint Joseph's Hawks": "Saint Joseph's",
    "St. Joseph's Hawks": "Saint Joseph's",
    "Saint Louis Billikens": "Saint Louis",
    "St. Louis Billikens": "Saint Louis",
    "UL Monroe Warhawks": "Louisiana Monroe",
    "Louisiana-Monroe Warhawks": "Louisiana Monroe",
    "UL Lafayette Ragin' Cajuns": "Louisiana",
    "Louisiana-Lafayette Ragin' Cajuns": "Louisiana",
    "Cal State Fullerton Titans": "CS Fullerton",
    "Cal State Bakersfield Roadrunners": "CS Bakersfield",
    "FIU Panthers": "Florida International",
    "Middle Tennessee Blue Raiders": "Middle Tennessee",
    "Mississippi State Bulldogs": "Mississippi State",
    "Kansas State Wildcats": "Kansas State",
    "Oklahoma State Cowboys": "Oklahoma State",
    "Boise State Broncos": "Boise State",
    "Colorado State Rams": "Colorado State",
    "Fresno State Bulldogs": "Fresno State",
    "San Diego State Aztecs": "San Diego State",
    "Utah State Aggies": "Utah State",
    "Nevada Wolf Pack": "Nevada",
    "Brigham Young Cougars": "BYU",
}


def build_reasons(result, team1, team2):
    reasons = []
    sig = result["signals"]

    if sig["net_rating_diff"] > 0:
        reasons.append(f"{team1} has the stronger net rating.")
    else:
        reasons.append(f"{team2} has the stronger net rating.")

    if sig["off_diff"] > 0:
        reasons.append(f"{team1} projects better offensively.")
    if sig["def_diff"] > 0:
        reasons.append(f"{team1} projects better defensively.")
    if sig["sos_diff"] > 0:
        reasons.append(f"{team1} has faced the tougher schedule.")

    return reasons


def render_top_list(title, df, value_column):
    st.write(f"**{title}**")
    if df.empty:
        st.write("- None")
        return
    for _, row in df.head(3).iterrows():
        st.write(
            f"- {row['Bet Side']} | {row['Team 1']} vs {row['Team 2']} | "
            f"Bet-side edge {row['Bet Edge']:+.1f} | Edge {row['Edge %']:+.1f}% | {row[value_column]}"
        )


def render_metric_row(label, value):
    st.write(f"**{label}:** {value}")


def render_scanner_card(row):
    title = f"{row['Recommendation']} - {row['Bet Side'] or 'PASS'}"
    with st.container(border=True):
        st.write(f"**{title}**")
        st.write(f"{row['Team 1']} vs {row['Team 2']}")
        render_metric_row("Spread", f"{row['Spread']:+.1f}")
        render_metric_row("Projected margin", f"{row['Projected margin']:+.1f}")
        render_metric_row("Bet-side edge", f"{row['Bet Edge']:+.1f}")
        render_metric_row("Edge", f"{row['Edge %']:+.1f}%")
        render_metric_row("Predicted winner", row["Predicted winner"])
        if row.get("Review"):
            render_metric_row("Review", row["Review"])


def get_live_label(edge):
    abs_edge = abs(edge)
    if abs_edge >= 4:
        return "Best edge"
    if abs_edge >= 2:
        return "Small lean"
    return "Pass"


def normalize_live_team_name(name, team_lookup):
    mapped_name = TEAM_NAME_MAP.get(name, name)
    if mapped_name in team_lookup:
        return mapped_name

    def canonicalize(value):
        value = str(value).replace("&", "and").replace("St.", "State").strip().lower()
        value = re.sub(r"[^a-z0-9 ]", "", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    normalized = canonicalize(mapped_name)
    team_lookup_map = {canonicalize(team): team for team in team_lookup}

    if normalized in team_lookup_map:
        return team_lookup_map[normalized]

    normalized_tokens = normalized.split()
    for token_count in range(len(normalized_tokens) - 1, 0, -1):
        candidate = " ".join(normalized_tokens[:token_count])
        if candidate in team_lookup_map:
            return team_lookup_map[candidate]

    for canonical_team_name, original_team_name in team_lookup_map.items():
        if canonical_team_name in normalized or normalized in canonical_team_name:
            return original_team_name

    return None


def get_bet_edge(spread_edge, bet_side):
    if not bet_side or spread_edge is None:
        return 0.0
    return abs(float(spread_edge))


def get_review_flag(edge_points):
    if edge_points >= 10:
        return "Extreme edge - manual review"
    if edge_points >= 8:
        return "Large edge - review"
    return ""


def format_edge_display(edge_points, cap=10.0):
    value = float(edge_points)
    if abs(value) >= cap:
        sign = "+" if value >= 0 else "-"
        return f"{sign}{cap:.0f}+"
    return f"{value:+.1f}"


def fetch_live_odds(api_key):
    params = urlencode({
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,h2h",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel",
    })
    url = f"https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds?{params}"
    try:
        with urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except ssl.SSLCertVerificationError:
        insecure_context = ssl._create_unverified_context()
        with urlopen(url, timeout=10, context=insecure_context) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc):
            insecure_context = ssl._create_unverified_context()
            with urlopen(url, timeout=10, context=insecure_context) as response:
                return json.loads(response.read().decode("utf-8"))
        raise


@st.cache_data(ttl=300)
def get_cached_live_odds(api_key):
    return fetch_live_odds(api_key)


def extract_market_spread(game):
    for bookmaker in game.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "spreads":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) < 2:
                continue
            spreads = {outcome.get("name"): outcome.get("point") for outcome in outcomes}
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            if home_team in spreads and away_team in spreads:
                return bookmaker.get("title"), away_team, spreads[away_team]
    return None, None, None


def build_live_bets(odds_games, team_lookup):
    rows = []
    skipped = []
    for game in odds_games:
        book_name, team1_market_name, spread = extract_market_spread(game)
        if spread is None:
            continue
        away_team = game.get("away_team")
        home_team = game.get("home_team")
        team1_name = normalize_live_team_name(away_team, team_lookup)
        team2_name = normalize_live_team_name(home_team, team_lookup)
        if team1_name is None or team2_name is None:
            skipped.append(f"{away_team} vs {home_team}")
            continue
        try:
            result = analyze_matchup(team1_name, team2_name, float(spread))
        except ValueError:
            skipped.append(f"{away_team} vs {home_team}")
            continue
        edge = abs(result["spread_edge"])
        side = result["bet_side"] or "Pass"
        review_flag = get_review_flag(edge)
        rows.append({
            "Matchup": f"{team1_name} vs {team2_name}",
            "Team 1": team1_name,
            "Team 2": team2_name,
            "Sportsbook": book_name,
            "Current line": f"{team1_name} {float(spread):+.1f}",
            "Model line": f"{team1_name} {result['model_spread']:+.1f}",
            "Edge": edge,
            "Raw spread edge": result["spread_edge"],
            "Absolute edge": edge,
            "Label": get_live_label(result["spread_edge"]),
            "Bet Side": side,
            "Review": review_flag,
        })
    return pd.DataFrame(rows), skipped


def render_live_bet_card(row, hero=False):
    title = "Best Right Now" if hero else row["Label"]
    with st.container(border=True):
        st.subheader(title)
        st.write(f"**{row['Bet Side']}**")
        st.write(row["Matchup"])
        render_metric_row("Current line", row["Current line"])
        render_metric_row("Model line", row["Model line"])
        render_metric_row("Edge", format_edge_display(row["Edge"]))
        render_metric_row("Book", row["Sportsbook"] or "N/A")
        if row.get("Review"):
            render_metric_row("Review", row["Review"])


def render_section_intro(title, body):
    st.markdown(f"### {title}")
    st.caption(body)


def get_optional_secret(name):
    try:
        return st.secrets[name]
    except (KeyError, StreamlitSecretNotFoundError):
        return None


st.set_page_config(page_title="March Madness Betting Helper", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: calc(env(safe-area-inset-top, 0px) + 1.1rem);
        padding-bottom: 4rem;
        max-width: 760px;
    }
    h1, h2, h3 {
        margin-top: 0;
        padding-top: 0;
    }
    div[data-testid="stTabs"] {
        margin-top: 0.6rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
    }
    button[kind="primary"] {
        width: 100%;
        min-height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

teams = get_team_list()

if "live_bets_df" not in st.session_state:
    st.session_state["live_bets_df"] = None
if "live_skipped_games" not in st.session_state:
    st.session_state["live_skipped_games"] = []
if "live_bets_error" not in st.session_state:
    st.session_state["live_bets_error"] = None
if "session_odds_api_key" not in st.session_state:
    st.session_state["session_odds_api_key"] = ""

live_tab, single_game_tab, scanner_tab = st.tabs(["Today's Vibe Bets", "Single Game", "Scanner"])

st.markdown("## 🏀 March Madness Bets")
st.caption("Quick picks — for fun only")

with live_tab:
    render_section_intro("Today's Vibe Bets", "See the posted NCAAB board, rank the best edges, and decide fast.")
    configured_api_key = get_optional_secret("THE_ODDS_API_KEY")

    if configured_api_key:
        api_key = configured_api_key
    else:
        pasted_api_key = st.text_input(
            "Paste Odds API key for this session",
            value=st.session_state["session_odds_api_key"],
            type="password",
            key="session_odds_api_key_input",
        )
        save_api_key = st.button("Use This Key", key="save_session_odds_api_key")
        clear_api_key = st.button("Clear Key", key="clear_session_odds_api_key")
        if save_api_key:
            st.session_state["session_odds_api_key"] = pasted_api_key.strip()
            st.session_state["live_bets_df"] = None
            st.session_state["live_skipped_games"] = []
            st.session_state["live_bets_error"] = None
        if clear_api_key:
            st.session_state["session_odds_api_key"] = ""
            st.session_state["live_bets_df"] = None
            st.session_state["live_skipped_games"] = []
            st.session_state["live_bets_error"] = None
        api_key = st.session_state["session_odds_api_key"]

    refresh_live = st.button("Refresh Board", key="refresh_live_bets")

    if not api_key:
        st.info("Add `THE_ODDS_API_KEY` to Streamlit secrets or paste an Odds API key above for this session.")
    else:
        if refresh_live or st.session_state["live_bets_df"] is None:
            try:
                get_cached_live_odds.clear()
                odds_games = get_cached_live_odds(api_key)
                live_bets_df, skipped_games = build_live_bets(odds_games, teams)
                st.session_state["live_bets_df"] = live_bets_df.sort_values(by="Absolute edge", ascending=False) if not live_bets_df.empty else live_bets_df
                st.session_state["live_skipped_games"] = skipped_games
                st.session_state["live_bets_error"] = None
            except Exception as exc:
                st.session_state["live_bets_error"] = str(exc)

        if st.session_state["live_bets_error"]:
            st.error(f"Live Bets failed to load: {st.session_state['live_bets_error']}")
        elif st.session_state["live_bets_df"] is None:
            st.info("Tap Refresh Board to load the current posted NCAAB board.")
        elif st.session_state["live_bets_df"].empty:
            st.warning("No posted NCAAB odds matched your current team names right now.")
        else:
            top_live = st.session_state["live_bets_df"].head(3).reset_index(drop=True)
            render_live_bet_card(top_live.iloc[0], hero=True)
            st.caption("Top card = strongest edge on the current posted board.")
            for _, row in top_live.iloc[1:].iterrows():
                render_live_bet_card(row)
            with st.expander("All Games"):
                st.dataframe(
                    st.session_state["live_bets_df"][["Matchup", "Bet Side", "Current line", "Model line", "Edge", "Label", "Sportsbook"]],
                    use_container_width=True,
                )
            if st.session_state["live_skipped_games"]:
                with st.expander("Skipped games"):
                    for game in st.session_state["live_skipped_games"]:
                        st.write(f"- {game}")

with single_game_tab:
    render_section_intro("Single Game", "Pick two teams, add a spread if you have one, and get a quick read.")

    team_col1, team_col2 = st.columns(2)
    with team_col1:
        team1 = st.selectbox("Team 1", teams, index=0, key="single_game_team1")
    with team_col2:
        team2 = st.selectbox("Team 2", teams, index=1 if len(teams) > 1 else 0, key="single_game_team2")

    spread_input = st.text_input(
        "Spread (Team 1)",
        value="",
        key="single_game_spread_input",
        placeholder="-1.5 or +1.5",
    )

    if st.button("Run Prediction", key="single_game_run_prediction"):
        if team1 == team2:
            st.error("Pick two different teams.")
        else:
            spread = None
            if spread_input.strip():
                try:
                    spread = float(spread_input)
                except ValueError:
                    st.warning("Spread must be a number like -1.5 or +1.5.")

            result = analyze_matchup(team1, team2, spread)

            with st.container(border=True):
                st.subheader("Quick Read")
                st.write(f"**{result['recommendation'] if spread is not None else 'Prediction'}**")
                st.write(f"{result['predicted_winner']} is the model pick")
                render_metric_row(f"{team1} win %", f"{result['team1_prob']:.1%}")
                render_metric_row(f"{team2} win %", f"{result['team2_prob']:.1%}")
                render_metric_row("Confidence", f"{result['confidence']:.1%}")
                render_metric_row("Projected margin", f"{result['projected_margin']:+.1f}")

            if spread is not None:
                with st.container(border=True):
                    st.subheader("Bet View")
                    render_metric_row("Bet side", result["bet_side"] or "PASS")
                    render_metric_row("Recommendation", result["recommendation"])
                    render_metric_row("Market win %", f"{result['market_prob']:.1%}")
                    render_metric_row("Probability edge", f"{result['edge']:+.1%}")
                    render_metric_row("Market spread", f"{result['spread']:+.1f}")
                    render_metric_row("Model spread", f"{result['model_spread']:+.1f}")
                    render_metric_row("Spread edge", f"{result['spread_edge']:+.1f}")
            else:
                st.info("Enter a spread to compare the model against the market.")

            with st.expander("Why the model likes this matchup"):
                for reason in build_reasons(result, team1, team2):
                    st.write(f"- {reason}")

with scanner_tab:
    render_section_intro("Scanner", "Upload a matchup CSV, add spreads, and rank the board by edge.")

    hide_big_spreads = st.checkbox("Hide spreads above 15", value=False, key="scanner_hide_big_spreads")

    uploaded_file = st.file_uploader("Upload matchup CSV", type=["csv"], key="scanner_upload_matchup_csv")

    if uploaded_file is not None:
        matchups = pd.read_csv(uploaded_file)
        matchups.columns = [str(col).strip().lower() for col in matchups.columns]

        required_columns = {"team1", "team2"}
        if not required_columns.issubset(matchups.columns):
            st.error("Matchup CSV must include columns named team1 and team2.")
        else:
            st.caption("Optional CSV column: spread")
            scanner_rows = []

            for idx, row in matchups.reset_index(drop=True).iterrows():
                st.markdown(f"**{row['team1']} vs {row['team2']}**")
                default_spread = ""
                if "spread" in matchups.columns and pd.notna(row.get("spread")):
                    default_spread = str(row["spread"])
                spread_value = st.text_input(
                    f"Spread for {row['team1']} vs {row['team2']}",
                    value=default_spread,
                    key=f"spread_{idx}"
                )

                scanner_rows.append({
                    "team1": row["team1"],
                    "team2": row["team2"],
                    "spread_input": spread_value,
                })

            if st.button("Scan Board", key="scanner_scan_board"):
                results = []
                errors = []

                for row in scanner_rows:
                    if not str(row["spread_input"]).strip():
                        errors.append(f"Missing spread for {row['team1']} vs {row['team2']}.")
                        continue

                    try:
                        spread = float(row["spread_input"])
                        if hide_big_spreads and abs(spread) > 15:
                            continue
                        result = analyze_matchup(row["team1"], row["team2"], spread)
                        bet_edge = get_bet_edge(result["spread_edge"], result["bet_side"])
                        results.append({
                            "Team 1": row["team1"],
                            "Team 2": row["team2"],
                            "Spread": spread,
                            "Predicted winner": result["predicted_winner"],
                            "Bet Side": result["bet_side"],
                            "Bet Edge": bet_edge,
                            "Team 1 win %": result["team1_prob"],
                            "Market win %": result["market_prob"],
                            "Edge %": result["edge"],
                            "Recommendation": result["recommendation"],
                            "Projected margin": result["projected_margin"],
                            "Model spread": result["model_spread"],
                            "Spread edge": result["spread_edge"],
                            "Review": get_review_flag(bet_edge),
                        })
                    except ValueError as exc:
                        errors.append(f"{row['team1']} vs {row['team2']}: {exc}")
                    except FileNotFoundError as exc:
                        errors.append(str(exc))

                for error in errors:
                    st.warning(error)

                if results:
                    results_df = pd.DataFrame(results)
                    results_df["Edge %"] = results_df["Edge %"] * 100
                    results_df["Spread edge rank"] = results_df["Bet Edge"]
                    results_df["Edge rank"] = results_df["Edge %"].abs()
                    results_df = results_df.sort_values(
                        by=["Spread edge rank", "Edge rank"],
                        ascending=False
                    ).drop(columns=["Spread edge rank", "Edge rank"])

                    top_bets = results_df[results_df["Recommendation"].isin(["BET", "STRONG BET"])]
                    top_leans = results_df[results_df["Recommendation"] == "LEAN"]
                    top_underdogs = results_df[
                        ((results_df["Spread"] > 0) & (results_df["Bet Side"] == results_df["Team 1"] + " ATS"))
                        | ((results_df["Spread"] < 0) & (results_df["Bet Side"] == results_df["Team 2"] + " ATS"))
                    ]

                    with st.container(border=True):
                        render_top_list("Top 3 Bets", top_bets, "Recommendation")
                    with st.container(border=True):
                        render_top_list("Top 3 Leans", top_leans, "Recommendation")
                    with st.container(border=True):
                        render_top_list("Top 3 Underdog Spots", top_underdogs, "Recommendation")

                    st.subheader("Quick Picks")
                    for _, row in results_df.head(3).iterrows():
                        render_scanner_card(row)

                    formatters = {
                        "Team 1 win %": "{:.1%}",
                        "Market win %": "{:.1%}",
                        "Edge %": "{:.1f}%",
                        "Bet Edge": "{:+.1f}",
                        "Spread": "{:+.1f}",
                        "Projected margin": "{:+.1f}",
                        "Model spread": "{:+.1f}",
                        "Spread edge": "{:+.1f}",
                    }
                    with st.expander("See full board table"):
                        st.dataframe(results_df.style.format(formatters), use_container_width=True)
    else:
        st.info("Upload a CSV with team1 and team2 columns to scan the board.")
