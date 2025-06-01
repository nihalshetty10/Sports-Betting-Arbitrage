import pandas as pd
from fair_odds_converter import prob_over, prob_to_american

def detect_arbitrage(df, threshold=0.546):
    # df should have: player, prop_type, line, predicted_mean, predicted_std, odds
    df['model_prob_over'] = prob_over(df['line'], df['predicted_mean'], df['predicted_std'])
    df['edge'] = df['model_prob_over'] - 0.546  # PrizePicks implied prob
    df['value_bet'] = df['edge'] > 0
    return df[df['value_bet']]

# Example usage:
# value_bets = detect_arbitrage(df)
