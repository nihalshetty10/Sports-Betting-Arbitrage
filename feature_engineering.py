import pandas as pd

def build_features(game_logs):
    # Example: rolling averages, weighted averages, etc.
    game_logs = game_logs.sort_values('GAME_DATE')
    game_logs['points_rolling_5'] = game_logs['PTS'].rolling(5, 1).mean()
    # Add more features as needed
    return game_logs
