import pandas as pd
import numpy as np
import os
import itertools

script_dir = os.path.dirname(os.path.abspath(__file__))
parlay_path = os.path.join(script_dir, "parlay_results.csv")
arb_path = os.path.join(script_dir, "arbitrage_ranked_props.csv")

# Load data
parlays = pd.read_csv(parlay_path)
arb = pd.read_csv(arb_path)

# PrizePicks payout structure
PAYOUTS = {
    3: {3: 3, 2: 1},
    4: {4: 6, 3: 1.5},
    5: {5: 10, 4: 2, 3: 0.4},
    6: {6: 25, 5: 2, 4: 0.4},
}

# Parlay EV calculation (all payout tiers)
def parlay_ev(probs, payout_dict):
    n = len(probs)
    ev = 0
    for hits, payout in payout_dict.items():
        k = hits
        prob = 0
        for hit_indices in itertools.combinations(range(n), k):
            p = 1
            for i in range(n):
                if i in hit_indices:
                    p *= probs[i]
                else:
                    p *= (1 - probs[i])
            prob += p
        ev += prob * payout
    return ev

# Probability of getting any payout (any win, not just all legs)
def parlay_your_win_prob(probs, payout_dict):
    n = len(probs)
    win_prob = 0
    for hits, payout in payout_dict.items():
        if payout > 0:
            k = hits
            prob = 0
            for hit_indices in itertools.combinations(range(n), k):
                p = 1
                for i in range(n):
                    if i in hit_indices:
                        p *= probs[i]
                    else:
                        p *= (1 - probs[i])
                prob += p
            win_prob += prob
    return win_prob

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

# Helper: get model_prob for each leg from arbitrage_ranked_props.csv
def get_model_probs(players, props):
    probs = []
    for player, prop in zip(players, props):
        row = arb[(arb['player'] == player) & (arb['prop_type'] == prop)]
        if not row.empty and not pd.isna(row.iloc[0]['model_prob']):
            probs.append(float(row.iloc[0]['model_prob']))
        else:
            probs.append(0.5)  # fallback if missing
    return probs

# Helper: get line for each leg from arbitrage_ranked_props.csv
def get_lines(players, props):
    lines = []
    for player, prop in zip(players, props):
        row = arb[(arb['player'] == player) & (arb['prop_type'] == prop)]
        if not row.empty and not pd.isna(row.iloc[0]['line']):
            lines.append(str(row.iloc[0]['line']))
        else:
            lines.append('N/A')
    return lines

# Kelly formula for parlays
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
    parlay_ev_val = row['parlay_ev']
    total_edge = row['total_edge']
    players = row['parlay_players'].split('|')
    props = row['parlay_props'].split('|')
    edges = [float(e) for e in row['parlay_edges'].split('|')]
    picks = get_picks(players, props)
    model_probs = get_model_probs(players, props)
    lines = get_lines(players, props)
    # Probability of getting any payout (any win)
    your_win_prob = parlay_your_win_prob(model_probs, PAYOUTS[leg_size])
    # Use main payout for Kelly (all legs hit)
    main_payout = {3: 3, 4: 6, 5: 10, 6: 25}[leg_size]
    # Use full payout structure for EV
    ev = parlay_ev(model_probs, PAYOUTS[leg_size])
    # Kelly fraction (capped between 0.25 and 0.5), now using your_win_prob
    kelly = kelly_fraction(ev, your_win_prob, main_payout)
    kelly_frac = min(max(kelly, 0.25), 0.5)
    bet_size = round(portfolio * kelly_frac, 2)
    portfolio_after_bet = round(portfolio - bet_size, 2)
    bet_rows.append({
        'parlay_id': parlay_id,
        'leg_size': leg_size,
        'parlay_ev': round(ev, 4),
        'total_edge': total_edge,
        'bet_size': bet_size,
        'picks': '|'.join(picks),
        'parlay_players': '|'.join(players),
        'parlay_props': '|'.join(props),
        'parlay_lines': '|'.join(lines),
        'your_win_prob': your_win_prob,
        'portfolio_after_bet': portfolio_after_bet
    })
    portfolio = portfolio_after_bet

bet_df = pd.DataFrame(bet_rows)
# Sort by your_win_prob descending
bet_df = bet_df.sort_values(by='your_win_prob', ascending=False)
bet_df.to_csv(os.path.join(script_dir, "auto_bet_log.csv"), index=False)
print(f"✅ Auto bet log saved to auto_bet_log.csv. Final portfolio: ${portfolio:.2f}") 
