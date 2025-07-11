import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error
import logging
import os

# Config
STAT_MAP = {
    "Hits": "H",
    "Total Bases": "TB",
    "Runs": "R",
    "Pitcher Strikeouts": "SO",
    "Hitter Strikeouts": "SO",
    "Earned Runs Allowed": "earned_runs"
}
DATA_FILE = "auto_bet_log.csv"

# Logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Load Parlay Data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = [col.strip() for col in df.columns]
    return df

# Load game stats for today's date
def load_game_stats():
    game_date = pd.to_datetime("today").normalize()
    fname = f"game_stats_{game_date.strftime('%Y-%m-%d')}.csv"
    if not os.path.exists(fname):
        logging.warning(f"Missing game stats file: {fname}")
        return pd.DataFrame()
    df = pd.read_csv(fname)
    df.columns = df.columns.str.strip()
    df['player'] = df['player'].str.strip()
    return df, game_date

# Evaluate Parlays using pre-scraped data
def evaluate_parlays(df, stats_df, game_date):
    result_rows = []

    for _, row in df.iterrows():
        players = row['parlay_players'].split('|')
        props = row['parlay_props'].split('|')
        picks = row['picks'].split('|')
        lines = list(map(float, row['parlay_lines'].split('|')))

        leg_results = []
        leg_hits = 0
        invalid = False

        for player, prop, pick, line in zip(players, props, picks, lines):
            stat_col = STAT_MAP.get(prop)
            stat_row = stats_df[stats_df['player'].str.lower() == player.strip().lower()]
            if stat_row.empty or stat_col not in stat_row.columns:
                leg_results.append("unknown")
                invalid = True
                continue

            try:
                actual = float(stat_row[stat_col].values[0])
            except:
                leg_results.append("unknown")
                invalid = True
                continue

            if (pick == "More" and actual > line) or (pick == "Less" and actual < line):
                leg_results.append("hit")
                leg_hits += 1
            else:
                leg_results.append("miss")

        leg_size = int(row['leg_size'])
        payout = 0
        profit = 0

        if not invalid:
            if leg_size == 3:
                payout = 3 if leg_hits == 3 else 1 if leg_hits == 2 else 0
            elif leg_size == 4:
                payout = 6 if leg_hits == 4 else 1.5 if leg_hits == 3 else 0
            elif leg_size == 5:
                payout = 10 if leg_hits == 5 else 2 if leg_hits == 4 else 0.4 if leg_hits == 3 else 0
            elif leg_size == 6:
                payout = 25 if leg_hits == 6 else 2 if leg_hits == 5 else 0.4 if leg_hits == 4 else 0
            profit = row['bet_size'] * payout - row['bet_size']

        result_rows.append({
            "date": game_date,
            "parlay_id": row['parlay_id'],
            "hits": leg_hits,
            "legs": leg_size,
            "leg_results": '|'.join(leg_results),
            "invalid": invalid,
            "profit": profit if not invalid else None,
            "your_win_prob": row['your_win_prob'] if not invalid else None
        })

    return pd.DataFrame(result_rows)

# Metrics Function
def calculate_metrics(results_df):
    results_df = results_df[results_df['invalid'] == False].copy()
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.sort_values(by='date').reset_index(drop=True)

    rolling_periods = {
        'Daily': 1,
        '3-Day': 3,
        'Weekly': 7,
        'Biweekly': 14,
        'Monthly': 30,
        'Quarterly': 90,
        'Yearly': 365
    }

    metrics = []
    for label, days in rolling_periods.items():
        for start in range(len(results_df)):
            window = results_df[(results_df['date'] >= results_df['date'].iloc[start]) &
                                (results_df['date'] < results_df['date'].iloc[start] + pd.Timedelta(days=days))]
            if len(window) < 2:
                continue
            profits = window['profit'].values
            win_pct = (profits > 0).mean()
            total_profit = profits.sum()
            sharpe = np.mean(profits) / (np.std(profits) + 1e-8)
            r2 = r2_score(results_df['hits'] / results_df['legs'], results_df['your_win_prob'])
            mae = mean_absolute_error(results_df['hits'] / results_df['legs'], results_df['your_win_prob'])
            max_drawdown = np.max(np.maximum.accumulate(np.cumsum(profits)) - np.cumsum(profits))

            metrics.append({
                "Period": label,
                "Start Date": window['date'].iloc[0].date(),
                "End Date": window['date'].iloc[-1].date(),
                "Total Profit": round(total_profit, 2),
                "Profit per Bet": round(total_profit / len(window), 2),
                "Win %": round(win_pct * 100, 2),
                "Sharpe Ratio": round(sharpe, 4),
                "Max Drawdown": round(max_drawdown, 2),
                "R^2": round(r2, 4),
                "MAE": round(mae, 4)
            })

    return pd.DataFrame(metrics)

# Main
if __name__ == "__main__":
    df = load_data()
    stats_df, game_date = load_game_stats()
    if stats_df.empty:
        print("⚠️ No stats file found for today. Exiting.")
    else:
        results = evaluate_parlays(df, stats_df, game_date)
        results.to_csv("evaluated_bets.csv", index=False)
        print("✅ Saved evaluated_bets.csv")

        metrics = calculate_metrics(results)
        metrics.to_csv("performance_metrics.csv", index=False)
        print("📈 Saved performance_metrics.csv")
