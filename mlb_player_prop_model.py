#pip install pybaseball
import pandas as pd
import pybaseball
from pybaseball import batting_stats, statcast_batter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests

# 1. Get player list for 2023 and 2024 (using MLB Stats API - fetching season stats)
player_ids = set() # Use a set to store unique player IDs

try:
    for year in [2023, 2024]:
        # Endpoint to get season hitting stats
        season_stats_url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season={year}"
        print(f"Fetching season stats for {year} from: {season_stats_url}") # Debug print
        response = requests.get(season_stats_url)
        response.raise_for_status() # Raise an exception for bad status codes
        season_stats_data = response.json()

        # Assuming the player data is under 'stats' -> 'splits' -> 'player' -> 'id'
        if season_stats_data and 'stats' in season_stats_data and len(season_stats_data['stats']) > 0 and 'splits' in season_stats_data['stats'][0]:
            for entry in season_stats_data['stats'][0]['splits']:
                player_info = entry.get('player')
                if player_info and 'id' in player_info:
                    player_ids.add(player_info['id'])

except requests.exceptions.RequestException as e:
    print(f"Error fetching season stats: {e}")
    player_ids = set() # Reset to empty if fetching fails

player_ids = list(player_ids) # Convert set to list for iteration
print(f"Found {len(player_ids)} unique player IDs from 2023 and 2024 season stats.") # Debug print

# 3. Prepare data (using MLB Stats API game logs for the fetched player IDs)
stats = ['runs', 'total_bases', 'hits', 'home_runs', 'strikeouts', 'rbi']
# Mapping from our stat names to potential API field names (these might need adjustment)
api_stat_map = {
    'runs': 'runs', # API might use full names
    'total_bases': 'totalBases', # API might use 'totalBases'
    'hits': 'hits',
    'home_runs': 'homeRuns', # API might use 'homeRuns'
    'strikeouts': 'strikeOuts', # API might use 'strikeOuts'
    'rbi': 'rbi',
}

data = []
# We will iterate through the fetched player_ids
# Limiting to first 20 players for initial testing to speed things up
for player_id in player_ids[:20]:
    print(f"Processing player ID: {player_id}")
    try:
        # Attempt to get game logs for the player for 2023 and 2024
        all_logs = []
        for year in [2023, 2024]:
            # Endpoint pattern for game logs by player for a year:
            game_log_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&season={year}&group=hitting"
            print(f"Fetching game logs for player {player_id} ({year}) from: {game_log_url}") # Debug print
            response = requests.get(game_log_url)
            response.raise_for_status()
            player_stats_data = response.json()

            # Assuming the game logs are nested under 'stats' -> 'splits'
            player_logs = []
            if player_stats_data and 'stats' in player_stats_data and len(player_stats_data['stats']) > 0 and 'splits' in player_stats_data['stats'][0]:
                 player_logs = player_stats_data['stats'][0]['splits']

            # Convert list of dicts to DataFrame
            if player_logs:
                 game_data_list = []
                 for game in player_logs:
                      game_date = game.get('date')
                      game_stats = game.get('stat', {}) # Get the nested stat dictionary
                      game_data = {
                          'game_date': game_date,
                          'runs': game_stats.get('runs', 0), # Use .get with default 0, using full name now
                          'totalBases': game_stats.get('totalBases', 0), # Using full name now
                          'hits': game_stats.get('hits', 0),
                          'homeRuns': game_stats.get('homeRuns', 0), # Using full name now
                          'strikeOuts': game_stats.get('strikeOuts', 0), # Using full name now
                          'rbi': game_stats.get('rbi', 0),
                      }
                      game_data_list.append(game_data)
                 all_logs.append(pd.DataFrame(game_data_list))

        if not all_logs:
            print(f"No game logs found for player {player_id} across 2023 and 2024")
            continue # Skip to next player if no logs for either year

        logs = pd.concat(all_logs).dropna(subset=['game_date']) # Concatenate and drop rows without date

        if logs.empty or len(logs) < 11:
            print(f"Skipping player {player_id}: Not enough game logs ({len(logs)} found) across 2023 and 2024")
            continue

        # Ensure game_date is datetime for proper sorting
        logs['game_date'] = pd.to_datetime(logs['game_date'])
        logs = logs.sort_values('game_date').reset_index(drop=True)

        last_10 = logs.tail(10)
        historical = logs.iloc[:-10]

        features = {}
        # Using the full stat names now as keys based on the API structure assumption
        full_api_stat_map = {
             'runs': 'runs',
             'total_bases': 'totalBases',
             'hits': 'hits',
             'home_runs': 'homeRuns',
             'strikeouts': 'strikeOuts',
             'rbi': 'rbi',
        }

        for stat, api_key in full_api_stat_map.items():
             # Check if the API key exists in the fetched logs DataFrame columns
             last_10_avg = last_10[api_key].mean() if api_key in last_10.columns and not last_10.empty else 0
             hist_avg = historical[api_key].mean() if api_key in historical.columns and not historical.empty else 0
             features[f"{stat}_weighted"] = 0.65 * last_10_avg + 0.35 * hist_avg

        # Targets: next game's stats (which is the last game in the sorted logs)
        target_row = logs.iloc[-1] # This is actually the last game *in the fetched logs*, used as the target
        for stat, api_key in full_api_stat_map.items():
            # Use .get for safety
            features[f"target_{stat}"] = target_row.get(api_key, 0)

        data.append(features)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching game logs for player {player_id}: {e}")
    except Exception as e:
        print(f"Error processing data for player {player_id}: {e}")

df = pd.DataFrame(data)
print(df.head())

# 4. Train/test split for each stat
for stat in stats:
    X = df[[f'{s}_weighted' for s in stats]]
    y = df[f'target_{stat}']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'--- {stat.upper()} ---')
    print('MSE:', mean_squared_error(y_test, y_pred))

# 5. Create a prediction function
def predict_player_stats(player_id, trained_models, api_stat_map, stats):
    """Predicts the next game stats for a given player ID."""
    try:
        # Get all games for 2023 and 2024 for the specific player
        logs_23 = statcast_batter('2023-03-01', '2023-11-01', player_id)
        logs_24 = statcast_batter('2024-03-01', '2024-11-01', player_id)
        logs = pd.concat([logs_23, logs_24])

        if logs.empty or len(logs) < 11:
            print(f"Not enough game logs (need at least 11) for player {player_id}")
            return None

        logs = logs.sort_values('game_date')
        last_10 = logs.tail(10)
        historical = logs.iloc[:-10]

        features = {}
        for stat in stats:
            stat_col = api_stat_map[stat]
            last_10_avg = last_10[stat_col].mean() if stat_col in last_10.columns and not last_10.empty else 0
            hist_avg = historical[stat_col].mean() if stat_col in historical.columns and not historical.empty else 0
            features[f"{stat}_weighted"] = 0.65 * last_10_avg + 0.35 * hist_avg

        # Create a DataFrame for prediction (model expects 2D input)
        X_predict = pd.DataFrame([features])
        # Ensure columns match training data features
        X_train_cols = [f'{s}_weighted' for s in stats]
        X_predict = X_predict.reindex(columns=X_train_cols, fill_value=0)

        predictions = {}
        for stat in stats:
            model = trained_models[stat] # Get the model for the specific stat
            prediction = model.predict(X_predict)[0] # Predict and get the single value
            predictions[stat] = max(0, round(prediction)) # Predictions can't be negative

        return predictions

    except Exception as e:
        print(f"Error predicting for player {player_id}: {e}")
        return None

# 6. Train models for each stat and store them in a dictionary
trained_models = {}
for stat in stats:
    X = df[[f'{s}_weighted' for s in stats]]
    y = df[f'target_{stat}']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(random_state=42) # Add random_state for reproducibility
    model.fit(X_train, y_train)
    trained_models[stat] = model # Store the trained model
    y_pred = model.predict(X_test)
    print(f'--- {stat.upper()} ---')
    print('MSE:', mean_squared_error(y_test, y_pred))

# 7. Example usage: Predict stats for a specific player
# Replace 'some_player_id' with an actual player's Fangraphs ID (IDfg)
# You can find IDs in the 'players' DataFrame printed earlier or from Fangraphs
example_player_id = player_ids[0] if player_ids else None # Using the first player from the fetched list as an example, with a check if list is empty

if example_player_id:
    predicted_stats = predict_player_stats(example_player_id, trained_models, api_stat_map, stats)

    if predicted_stats:
        print(f"\nPredicted stats for player ID {example_player_id}:")
        for stat, pred_value in predicted_stats.items():
            print(f"  {stat.replace('_weighted', '').capitalize()}: {pred_value}") 