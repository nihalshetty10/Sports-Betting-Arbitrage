from nba_api.stats.endpoints import playergamelog
import pandas as pd

def get_player_gamelog(player_id, seasons=None):
    """
    Fetches game logs for the given player_id for the specified seasons.
    If seasons is None, defaults to the last two NBA seasons.
    """
    if seasons is None:
        # Update these as new seasons start!
        seasons = ["2022-23", "2023-24"]
    all_logs = []
    for season in seasons:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
        df['SEASON'] = season
        all_logs.append(df)
    return pd.concat(all_logs, ignore_index=True)

def get_last_n_games(player_id, n=10):
    df = get_player_gamelog(player_id)
    return df.head(n)

# Example usage:
# logs = get_player_gamelog(player_id='201939')  # Stephen Curry
# print(logs.head())

# Add more functions for team stats, pace, defense, etc.
