import pandas as pd
import numpy as np

# CONFIG
STAT_MAP = {
    "Hits": "H",
    "Total Bases": "TB",
    "Runs": "R",
    "Pitcher Strikeouts": "Pitcher SO",
    "Hitter Strikeouts": "Hitter SO",
    "Earned Runs Allowed": "ER"
}

PAYOUT_TABLE = {
    3: {3: 3, 2: 1},
    4: {4: 6, 3: 1.5},
    5: {5: 10, 4: 2, 3: 0.4},
    6: {6: 25, 5: 2, 4: 0.4}
}

# LOAD DATA
parlays_df = pd.read_csv("auto_bet_log.csv")
game_stats_df = pd.read_csv("game_stats_2025-07-11.csv")

# Normalize player names
def normalize(name):
    return name.strip()

game_stats_df["player"] = game_stats_df["player"].apply(normalize)

# Evaluate parlays
def evaluate_parlay(row):
    players = row["parlay_players"].split("|")
    stats = row["parlay_props"].split("|")
    lines = list(map(float, row["parlay_lines"].split("|")))
    picks = row["picks"].split("|")
    leg_size = row["leg_size"]
    bet_size = row["bet_size"]

    correct = 0
    leg_results = []
    matched_legs = 0

    for player, stat, line, pick in zip(players, stats, lines, picks):
        player = normalize(player)
        stat_col = STAT_MAP.get(stat)
        if stat_col not in game_stats_df.columns:
            leg_results.append("unknown")
            continue
        value_series = game_stats_df.loc[game_stats_df["player"] == player, stat_col]
        if value_series.empty:
            leg_results.append("unknown")
            continue
        actual = value_series.values[0]
        matched_legs += 1
        if pick == "More":
            hit = actual > line
        elif pick == "Less":
            hit = actual < line
        else:
            hit = False
        leg_results.append(str(hit))
        if hit:
            correct += 1

    invalid = matched_legs == 0
    payout_multiplier = PAYOUT_TABLE.get(leg_size, {}).get(correct, 0)
    profit = bet_size * (payout_multiplier - 1) if not invalid else 0.0

    return pd.Series([correct, leg_size, "|".join(leg_results), invalid, profit])

# Apply evaluation
parlays_df[["hits", "legs", "leg_results", "invalid", "profit"]] = parlays_df.apply(evaluate_parlay, axis=1)

# Metrics
valid = parlays_df[~parlays_df["invalid"]]
total_profit = valid["profit"].sum()
win_rate = (valid["profit"] > 0).mean()
sharpe = valid["profit"].mean() / valid["profit"].std() if valid["profit"].std() != 0 else np.nan

print(f"Total Profit/Loss: ${total_profit:.2f}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

# Save
parlays_df.to_csv("evaluated_bets.csv", index=False)
