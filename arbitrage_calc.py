import pandas as pd
from scipy.stats import norm
import numpy as np
import itertools

df = pd.read_csv("all_player_prop_results.csv")

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

def generate_parlays(df, leg_sizes=[3, 4, 5, 6]):
    # Only use top-ranked props (e.g., top 30 for efficiency)
    top_props = df.head(30).copy()
    # Each prop: (player, prop_type, arbitrage_edge)
    prop_tuples = list(top_props[['player', 'prop_type', 'arbitrage_edge']].itertuples(index=False, name=None))
    # To avoid duplicate parlays with same set of players, keep a set of frozen player sets
    seen_player_sets = set()
    results = {k: [] for k in leg_sizes}
    for k in leg_sizes:
        # Generate all k-combinations of props
        for combo in itertools.combinations(prop_tuples, k):
            players = [c[0] for c in combo]
            # Skip if duplicate player in parlay
            if len(set(players)) < k:
                continue
            player_set = frozenset(players)
            # Skip if this set of players has already been used in a smaller parlay
            if player_set in seen_player_sets:
                continue
            seen_player_sets.add(player_set)
            # Parlay as list of (player, prop_type, edge)
            parlay = list(combo)
            # Parlay edge: product of individual edges (proxy for likelihood)
            parlay_edge = np.prod([c[2] for c in combo])
            results[k].append((parlay, parlay_edge))
        # Sort and keep only top_n
        results[k] = sorted(results[k], key=lambda x: x[1], reverse=True)
    return results

df['arbitrage_edge'] = df.apply(calculate_arbitrage_edge, axis=1)
df['best_pick'] = df.apply(determine_best_pick, axis=1)

# Sort by edge value descending
df = df.sort_values(by="arbitrage_edge", ascending=False)

# Save to CSV
df.to_csv("arbitrage_ranked_props.csv", index=False)

print("✅ Arbitrage analysis saved to arbitrage_ranked_props.csv")

# --- Parlay Generator ---
parlay_results = generate_parlays(df, leg_sizes=[3, 4, 5, 6], top_n=5)
for k in [3, 4, 5, 6]:
    print(f"\nTop {len(parlay_results[k])} {k}-leg parlays:")
    for i, (parlay, edge) in enumerate(parlay_results[k], 1):
        print(f"{i}. Parlay edge: {edge:.4f}")
        for player, prop, parlay_edge in parlay:
            print(f"   - {player} | {prop} | edge: {parlay_edge}")
