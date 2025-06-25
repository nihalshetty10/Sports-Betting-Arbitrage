import pandas as pd
from scipy.stats import norm

df = pd.read_csv("scraped_prizepicks_props.csv")

# Temporary placeholder standard deviation (update with real values later)
std_dev = 1.0

def calculate_arbitrage_edge(row):
    predicted = row['predicted_value']
    line = row['line']
    z = (predicted - line) / std_dev
    prob = norm.cdf(z)
    edge = abs(prob - 0.5) * 2 
    return round(edge, 4)

df['arbitrage_edge'] = df.apply(calculate_arbitrage_edge, axis=1)
df = df.sort_values(by="arbitrage_edge", ascending=False)

# Save result to a new CSV
df.to_csv("arbitrage_ranked_props.csv", index=False)

print("✅ Arbitrage analysis saved to arbitrage_ranked_props.csv")
