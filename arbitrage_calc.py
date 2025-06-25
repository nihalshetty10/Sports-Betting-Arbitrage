import pandas as pd
from scipy.stats import norm

df = pd.read_csv("scraped_prizepicks_props.csv")

# Placeholder standard deviation (update later)
std_dev = 1.0

def calculate_arbitrage_edge(row):
    predicted = row['predicted_value']
    line = row['line']
    z = (predicted - line) / std_dev
    prob = norm.cdf(z)
    edge = abs(prob - 0.5) * 2  # Normalized to [0, 1]
    return round(edge, 4)

def determine_best_pick(row):
    return "More" if row['predicted_value'] > row['line'] else "Less"

df['arbitrage_edge'] = df.apply(calculate_arbitrage_edge, axis=1)
df['best_pick'] = df.apply(determine_best_pick, axis=1)

# Sort by edge value descending
df = df.sort_values(by="arbitrage_edge", ascending=False)

# Save to CSV
df.to_csv("arbitrage_ranked_props.csv", index=False)

print("✅ Arbitrage analysis saved to arbitrage_ranked_props.csv")
