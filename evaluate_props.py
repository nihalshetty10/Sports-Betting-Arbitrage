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

# Add date to each row (since no date column exists)
parlays_df['date'] = pd.to_datetime("2025-07-11")

# Normalize player names
def normalize(name):
    return name.strip()

game_stats_df["player"] = game_stats_df["player"].apply(normalize)

skipped_players = []

# Evaluate parlays
def evaluate_parlay(row):
    players = row["parlay_players"].split("|")
    stats = row["parlay_props"].split("|")
    lines = list(map(float, row["parlay_lines"].split("|")))
    picks = row["picks"].split("|")
    bet_size = row["bet_size"]
    original_leg_size = row["leg_size"]

    correct = 0
    leg_results = []
    matched_legs = 0

    for player, stat, line, pick in zip(players, stats, lines, picks):
        player = normalize(player)
        stat_col = STAT_MAP.get(stat)

        if stat_col not in game_stats_df.columns:
            leg_results.append("unknown")
            skipped_players.append((player, "Missing stat column"))
            continue

        value_series = game_stats_df.loc[game_stats_df["player"] == player, stat_col]
        if value_series.empty:
            leg_results.append("unknown")
            skipped_players.append((player, "Player not found"))
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

    adjusted_leg_size = matched_legs
    invalid = adjusted_leg_size < 2

    if original_leg_size == 3 and adjusted_leg_size == 2:
        payout_multiplier = 1
        profit = 0
    elif invalid:
        payout_multiplier = 0
        profit = 0
    else:
        payout_multiplier = PAYOUT_TABLE.get(adjusted_leg_size, {}).get(correct, 0)
        profit = bet_size * (payout_multiplier - 1)

    return pd.Series([correct, adjusted_leg_size, "|".join(leg_results), invalid, profit])

# Apply evaluation
parlays_df[["hits", "legs", "leg_results", "invalid", "profit"]] = parlays_df.apply(evaluate_parlay, axis=1)

# Clean leg_results by removing 'unknown'
def clean_leg_results(legs):
    return "|".join([leg for leg in legs.split("|") if leg in ["True", "False"]])

parlays_df["leg_results"] = parlays_df["leg_results"].apply(clean_leg_results)

# Save cleaned evaluated bets
parlays_df.to_csv("evaluated_bets.csv", index=False)

# Save skipped players (deduplicated)
pd.DataFrame(list(set(skipped_players)), columns=["player", "reason"]).to_csv("skipped_players.csv", index=False)

# Metrics Summary
valid = parlays_df[~parlays_df["invalid"]]
daily = valid.groupby(valid['date'].dt.date)

metrics_rows = []
for date, group in daily:
    profits = group['profit']
    sharpe = profits.mean() / (profits.std() + 1e-8)
    win_rate = (profits > 0).mean()
    total_profit = profits.sum()
    win_counts = group[group['profit'] > 0]['legs'].value_counts().to_dict()

    metrics_rows.append({
        'Period': 'Daily',
        'Date': date,
        'Total Profit': round(total_profit, 2),
        'Win Rate': round(win_rate, 4),
        'Sharpe Ratio': round(sharpe, 4),
        '3-Leg Wins': win_counts.get(3, 0),
        '4-Leg Wins': win_counts.get(4, 0),
        '5-Leg Wins': win_counts.get(5, 0),
        '6-Leg Wins': win_counts.get(6, 0)
    })

# Duplicate for Weekly and Monthly
for row in metrics_rows.copy():
    weekly = row.copy(); weekly['Period'] = 'Weekly'; metrics_rows.append(weekly)
    monthly = row.copy(); monthly['Period'] = 'Monthly'; metrics_rows.append(monthly)

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv("summary_metrics.csv", index=False)

print("✅ Done: Saved evaluated_bets.csv, skipped_players.csv, and summary_metrics.csv")
