import os
import pandas as pd
from scipy.stats import norm
import numpy as np
import itertools

PAYOUTS = {
    3: {3: 3, 2: 1},           # 3-pick: 3/3=3x, 2/3=1x
    4: {4: 6, 3: 1.5},         # 4-pick: 4/4=6x, 3/4=1.5x
    5: {5: 10, 4: 2, 3: 0.4},  # 5-pick: 5/5=10x, 4/5=2x, 3/5=0.4x
    6: {6: 25, 5: 2, 4: 0.4},  # 6-pick: 6/6=25x, 5/6=2x, 4/6=0.4x
}

# Book-implied win probabilities for main payout (all legs correct)
BOOK_IMPLIED_WIN_PROB = {
    3: 0.577,
    4: 0.551,
    5: 0.525,
    6: 0.505,
}

def calculate_arbitrage_edge(row):
    predicted = row['predicted_value']
    line = row['line']
    sd = row.get('player_actual_sd', np.nan)
    if pd.notna(sd) and sd > 0:
        z = (predicted - line) / sd
        prob = norm.cdf(z)
        edge = abs(prob - 0.5) * 2  # Normalized to [0, 1]
        return round(edge, 4)
    else:
        # Fallback: use absolute difference as edge
        return round(abs(predicted - line), 4)

def determine_best_pick(row):
    return "More" if row['predicted_value'] > row['line'] else "Less"

def parlay_ev(probs, payout_dict):
    n = len(probs)
    ev = 0
    from itertools import combinations
    for hits, payout in payout_dict.items():
        k = hits
        prob = 0
        for hit_indices in combinations(range(n), k):
            p = 1
            for i in range(n):
                if i in hit_indices:
                    p *= probs[i]
                else:
                    p *= (1 - probs[i])
            prob += p
        ev += prob * payout
    return ev

def generate_parlays_with_ev(df, leg_sizes=[3, 4, 5, 6], prop_pool=200):
    top_props = df.head(prop_pool).copy()
    prop_tuples = list(top_props[['player', 'prop_type', 'arbitrage_edge']].itertuples(index=False, name=None))
    results = {k: [] for k in leg_sizes}
    parlays_by_playerset = {k: [] for k in leg_sizes}
    for k in leg_sizes:
        for combo in itertools.combinations(prop_tuples, k):
            players = [c[0] for c in combo]
            if len(set(players)) < k:
                continue
            player_set = frozenset(players)
            # Overlap check: no more than floor(k/2) players in common with any existing parlay of same size
            too_much_overlap = False
            for existing in parlays_by_playerset[k]:
                overlap = len(player_set & existing)
                if overlap > k // 2:
                    too_much_overlap = True
                    break
            if too_much_overlap:
                continue
            parlays_by_playerset[k].append(player_set)
            parlay = list(combo)
            probs = [0.5 + min(max(c[2], 0), 1) / 2 for c in combo]
            payout_dict = PAYOUTS[k]
            ev = parlay_ev(probs, payout_dict)
            results[k].append((parlay, ev))
        results[k] = sorted(results[k], key=lambda x: x[1], reverse=True)
    return results

# Use script-relative paths for all file operations
script_dir = os.path.dirname(os.path.abspath(__file__))
all_props_path = os.path.join(script_dir, "all_player_prop_results.csv")
arb_ranked_path = os.path.join(script_dir, "arbitrage_ranked_props.csv")
parlay_results_path = os.path.join(script_dir, "parlay_results.csv")

df = pd.read_csv(all_props_path)
df['arbitrage_edge'] = df.apply(calculate_arbitrage_edge, axis=1)
df['best_pick'] = df.apply(determine_best_pick, axis=1)

# Sort by edge value descending
df = df.sort_values(by="arbitrage_edge", ascending=False)

# Remove rows where prediction_status contains 'Player Not Found'
df = df[~df['prediction_status'].str.contains('Player Not Found', na=False)]

# Save to CSV
df.to_csv(arb_ranked_path, index=False)

print(f"✅ Arbitrage analysis saved to {arb_ranked_path}")

# --- Parlay Generator with EV ---
parlay_results = generate_parlays_with_ev(df, leg_sizes=[3, 4, 5, 6], prop_pool=100)

# Collect all parlays for CSV
parlay_rows = []
for k in [3, 4, 5, 6]:
    print(f"\nTop {len(parlay_results[k])} {k}-leg parlays by EV:")
    for i, (parlay, ev) in enumerate(parlay_results[k], 1):
        print(f"{i}. Parlay EV: {ev:.4f}")
        players = [player for player, prop, edge in parlay]
        props = [prop for player, prop, edge in parlay]
        edges = [str(edge) for player, prop, edge in parlay]
        # Calculate your win probability (probability all legs hit)
        book_win_prob = BOOK_IMPLIED_WIN_PROB[k]
        your_probs = [min(book_win_prob * (1 + edge), 1.0) for player, prop, edge in parlay]
        your_win_prob = np.prod(your_probs)
        total_edge = ((your_win_prob - book_win_prob) / book_win_prob) * 100
        for player, prop, edge in parlay:
            print(f"   - {player} | {prop} | edge: {edge}")
        parlay_rows.append({
            'leg_size': k,
            'parlay_rank': i,
            'parlay_ev': round(ev, 4),
            'parlay_players': '|'.join(players),
            'parlay_props': '|'.join(props),
            'parlay_edges': '|'.join(edges),
            'total_edge': round(total_edge, 2)
        })

# Save all parlays to CSV
parlay_df = pd.DataFrame(parlay_rows)
parlay_df.to_csv(parlay_results_path, index=False)
print(f"✅ Parlay results saved to {parlay_results_path}")
