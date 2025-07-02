import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parlay_path = os.path.join(script_dir, "parlay_results.csv")
arb_path = os.path.join(script_dir, "arbitrage_ranked_props.csv")

# Load data
parlays = pd.read_csv(parlay_path)
arb = pd.read_csv(arb_path)

# Filter parlays by criteria
good_parlays = parlays[(parlays['parlay_ev'] >= 1.05) & (parlays['total_edge'] >= 0.03)].copy()
good_parlays = good_parlays.reset_index(drop=True)

# Helper: get pick for each leg from arbitrage_ranked_props.csv
def get_picks(players, props):
    picks = []
    for player, prop in zip(players, props):
        row = arb[(arb['player'] == player) & (arb['prop_type'] == prop)]
        if not row.empty:
            picks.append(row.iloc[0]['best_pick'])
        else:
            picks.append('N/A')
    return picks

# Kelly formula for parlays (approx):
def kelly_fraction(ev, win_prob, payout):
    b = payout - 1
    q = 1 - win_prob
    f = (ev * win_prob - q) / b if b > 0 else 0
    return max(0, min(f, 1))

portfolio = 100.0
bet_rows = []

for idx, row in good_parlays.iterrows():
    parlay_id = idx + 1
    leg_size = row['leg_size']
    parlay_ev = row['parlay_ev']
    total_edge = row['total_edge']
    players = row['parlay_players'].split('|')
    props = row['parlay_props'].split('|')
    edges = [float(e) for e in row['parlay_edges'].split('|')]
    picks = get_picks(players, props)
    # Use main payout for Kelly (e.g., 3x for 3-leg, 6x for 4-leg, etc.)
    main_payout = {3: 3, 4: 6, 5: 10, 6: 25}[leg_size]
    # Use your_win_prob as in previous script
    book_win_prob = {3: 0.577, 4: 0.551, 5: 0.525, 6: 0.505}[leg_size]
    your_probs = [min(book_win_prob * (1 + edge), 1.0) for edge in edges]
    your_win_prob = np.prod(your_probs)
    # Kelly fraction (capped between 0.25 and 0.5)
    kelly = kelly_fraction(parlay_ev, your_win_prob, main_payout)
    kelly_frac = min(max(kelly, 0.25), 0.5)
    bet_size = round(portfolio * kelly_frac, 2)
    portfolio_after_bet = round(portfolio - bet_size, 2)
    bet_rows.append({
        'parlay_id': parlay_id,
        'leg_size': leg_size,
        'parlay_ev': parlay_ev,
        'total_edge': total_edge,
        'bet_size': bet_size,
        'picks': '|'.join(picks),
        'portfolio_after_bet': portfolio_after_bet
    })
    portfolio = portfolio_after_bet

bet_df = pd.DataFrame(bet_rows)
bet_df.to_csv(os.path.join(script_dir, "auto_bet_log.csv"), index=False)
print(f"✅ Auto bet log saved to auto_bet_log.csv. Final portfolio: ${portfolio:.2f}") 