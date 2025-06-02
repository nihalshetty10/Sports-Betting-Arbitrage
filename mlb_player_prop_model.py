#pip install pybaseball
import pandas as pd
import pybaseball
from pybaseball import batting_stats, statcast_batter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
import time # Import time for rate limiting
import datetime # Import datetime to get current year

# Get current year dynamically
current_year = datetime.datetime.now().year
year_minus_1 = current_year - 1
year_minus_2 = current_year - 2
year_plus_0 = current_year # This will be 2025 if run in 2025

# Function to get player details including primary position
def get_player_details(player_id):
    details_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    try:
        response = requests.get(details_url)
        response.raise_for_status()
        details_data = response.json()
        if details_data and 'people' in details_data and len(details_data['people']) > 0:
            person_data = details_data['people'][0]
            return {
                'fullName': person_data.get('fullName', f'Player ID: {player_id}'),
                'primaryPosition': person_data.get('primaryPosition', {})
            }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching details for player {player_id}: {e}")
    return None

# 1. Get player list, names, and positions for the last two years (using MLB Stats API season stats)
player_info_map = {} # Dictionary to store player info (name, position)
player_ids = set() # Use a set to store unique player IDs

# Fetch initial player list from the last two *full* seasons for training data identification
training_years = [year_minus_1, year_minus_2]

try:
    # Fetch Hitting Season Stats for the last two full years
    for year in training_years:
        season_stats_url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season={year}"
        print(f"Fetching hitting season stats for {year} from: {season_stats_url}")
        response = requests.get(season_stats_url)
        response.raise_for_status()
        season_stats_data = response.json()

        if season_stats_data and 'stats' in season_stats_data and len(season_stats_data['stats']) > 0 and 'splits' in season_stats_data['stats'][0]:
            player_splits = season_stats_data['stats'][0]['splits']
            print(f"Found {len(player_splits)} hitting player entries (splits) in {year} season stats.")
            for entry in player_splits:
                player_info = entry.get('player')
                if player_info and 'id' in player_info:
                    player_id = player_info['id']
                    # Store player name and position, prioritizing hitting position if available
                    if player_id not in player_info_map or player_info_map[player_id].get('primaryPosition', {}).get('code') not in ['P', 'SP', 'RP']:
                         player_info_map[player_id] = {
                             'fullName': player_info.get('fullName', f'Player ID: {player_id}'),
                             'primaryPosition': player_info.get('primaryPosition', {})
                         }
                    player_ids.add(player_id)

    # Fetch Pitching Season Stats for the last two full years
    for year in training_years:
        season_stats_url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=pitching&season={year}"
        print(f"Fetching pitching season stats for {year} from: {season_stats_url}")
        response = requests.get(season_stats_url)
        response.raise_for_status()
        season_stats_data = response.json()

        if season_stats_data and 'stats' in season_stats_data and len(season_stats_data['stats']) > 0 and 'splits' in season_stats_data['stats'][0]:
            player_splits = season_stats_data['stats'][0]['splits']
            print(f"Found {len(player_splits)} pitching player entries (splits) in {year} season stats.")
            for entry in player_splits:
                player_info = entry.get('player')
                if player_info and 'id' in player_info:
                     player_id = player_info['id']
                     # Store player name and position, prioritizing pitching position if available
                     # This helps classify two-way players correctly if pitching is primary
                     player_info_map[player_id] = {
                          'fullName': player_info.get('fullName', f'Player ID: {player_id}'),
                          'primaryPosition': player_info.get('primaryPosition', {})
                     }
                     player_ids.add(player_id)

except requests.exceptions.RequestException as e:
    print(f"Error fetching season stats: {e}")
    player_ids = set() # Reset to empty if fetching fails
    player_info_map = {} # Reset mapping if fetching fails

player_ids = list(player_ids) # Convert set to list for iteration
print(f"Found {len(player_ids)} unique player IDs from {year_minus_2} and {year_minus_1} season stats for training.")

# Attempt to enrich player info with position if missing
print("Attempting to enrich player position information...")
updated_player_info_map = {}
for player_id in player_ids:
    if player_id not in player_info_map or not player_info_map[player_id].get('primaryPosition', {}).get('code'):
        # If position is missing from season stats, try fetching from player details endpoint
        details = get_player_details(player_id)
        if details:
             updated_player_info_map[player_id] = details
        else:
             # Keep existing incomplete info or create a minimal entry
             updated_player_info_map[player_id] = player_info_map.get(player_id, {'fullName': f'Player ID: {player_id}', 'primaryPosition': {}})
    else:
        # Keep existing complete info
        updated_player_info_map[player_id] = player_info_map[player_id]
    time.sleep(0.05) # Small delay for details API calls

player_info_map = updated_player_info_map # Update the main map

# 2. Define stats relevant to hitting and pitching
hitting_stats = ['runs', 'total_bases', 'hits', 'home_runs', 'rbi', 'strikeouts'] # Strikeouts as batter strikeouts
pitching_stats = ['strikeouts'] # Strikeouts as pitcher strikeouts

# Mapping from our stat names to potential API field names
api_hitting_stat_map = {
    'runs': 'runs',
    'total_bases': 'totalBases',
    'hits': 'hits',
    'home_runs': 'homeRuns',
    'rbi': 'rbi',
    'strikeouts': 'strikeOuts', # API key for batter strikeouts
}

api_pitching_stat_map = {
    'strikeouts': 'strikeOuts', # API key for pitcher strikeouts
    # Add other pitching stats here later if needed
}

# 3. Prepare data for TRAINING (using MLB Stats API game logs for 2023 and 2024)
# This will now build separate data for batters and pitchers for training
hitting_data_train = []
pitching_data_train = []

print("\n--- Preparing Training Data ({year_minus_2}-{year_minus_1}) ---")

for player_id in player_ids:
    player_info = player_info_map.get(player_id, {})
    primary_position_code = player_info.get('primaryPosition', {}).get('code', 'Unknown')
    is_pitcher = primary_position_code in ['P', 'SP', 'RP']

    # Only process players with known positions for training
    if primary_position_code == 'Unknown':
         # print(f"Skipping training data for player {player_info.get('fullName', f'ID: {player_id}')} (ID: {player_id}): Unknown position.")
         continue

    # print(f"Processing {player_info.get('fullName', f'ID: {player_id}')} (ID: {player_id}, Pos: {primary_position_code}) for training data.")

    try:
        all_logs = []
        # Determine which game logs to fetch based on primary position for training
        if is_pitcher:
            log_group = 'pitching'
            stats_to_process = pitching_stats
            api_stat_map_current = api_pitching_stat_map
        else:
            log_group = 'hitting'
            stats_to_process = hitting_stats
            api_stat_map_current = api_hitting_stat_map

        if not stats_to_process: # Skip if no relevant stats for this position
             # print(f"Skipping player {player_id}: No relevant stats defined for position {primary_position_code}")
             time.sleep(0.05)
             continue

        # Fetch game logs for the training years
        for year in training_years:
            game_log_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&season={year}&group={log_group}"
            response = requests.get(game_log_url)
            response.raise_for_status()
            player_stats_data = response.json()

            player_logs = []
            if player_stats_data and 'stats' in player_stats_data and len(player_stats_data['stats']) > 0 and 'splits' in player_stats_data['stats'][0]:
                 player_logs = player_stats_data['stats'][0]['splits']

            if player_logs:
                 game_data_list = []
                 for game in player_logs:
                      game_date = game.get('date')
                      game_stats = game.get('stat', {}) # Get the nested stat dictionary
                      game_data = {'game_date': game_date}
                      for stat_name, api_key in api_stat_map_current.items():
                          game_data[api_key] = game_stats.get(api_key, 0)
                      if all(api_key in game_stats for api_key in api_stat_map_current.values()):
                           game_data_list.append(game_data)
                 if game_data_list:
                      all_logs.append(pd.DataFrame(game_data_list))

        if not all_logs:
            # print(f"No valid {log_group} game logs found for training for player {player_id} across {year_minus_2} and {year_minus_1}")
            time.sleep(0.05)
            continue

        logs = pd.concat(all_logs).dropna(subset=['game_date']) # Concatenate and drop rows without date

        # Require at least two games to calculate features and have a target game for training
        if logs.empty or len(logs) < 2:
            # print(f"Skipping training data for player {player_id}: Not enough {log_group} game logs ({len(logs)} found) across {year_minus_2} and {year_minus_1} to calculate features/target")
            time.sleep(0.05)
            continue

        logs['game_date'] = pd.to_datetime(logs['game_date'])
        logs = logs.sort_values('game_date').reset_index(drop=True)

        # Features for training are based on all but the last game in the logs
        historical_logs = logs.iloc[:-1]

        # Calculate last 10 games from the most recent season in historical logs for training
        latest_year_in_logs = historical_logs['game_date'].dt.year.max()
        if latest_year_in_logs:
            last_10_logs = historical_logs[historical_logs['game_date'].dt.year == latest_year_in_logs].tail(10)
        else:
            last_10_logs = pd.DataFrame() # Empty if no latest year found (shouldn't happen with < 2 game check)

        features = {}
        # Calculate weighted features using the stats relevant to the position
        for stat_name, api_key in api_stat_map_current.items():
             last_10_avg = last_10_logs[api_key].mean() if api_key in last_10_logs.columns and not last_10_logs.empty else 0
             hist_avg = historical_logs[api_key].mean() if api_key in historical_logs.columns and not historical_logs.empty else 0
             features[f"{stat_name}_weighted"] = 0.75 * last_10_avg + 0.25 * hist_avg # Use 75/25 weighting for training

        # Targets: next game's stats (which is the last game in the sorted logs for training)
        target_row = logs.iloc[-1]
        for stat_name, api_key in api_stat_map_current.items():
            features[f"target_{stat_name}"] = target_row.get(api_key, 0)

        # Append to the correct data list
        if is_pitcher:
            pitching_data_train.append(features)
        else:
            hitting_data_train.append(features)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {log_group} game logs for training for player {player_id}: {e}")
    except Exception as e:
        print(f"Error processing training data for player {player_id}: {e}")

    time.sleep(0.05) # Add a small delay to avoid overwhelming the API

# Create separate DataFrames for hitting and pitching training data
hitting_df_train = pd.DataFrame(hitting_data_train)
pitching_df_train = pd.DataFrame(pitching_data_train)

print("\n--- Hitting Training Data Summary ---")
print("DataFrame head after data preparation:")
print(hitting_df_train.head())
print(f"DataFrame shape: {hitting_df_train.shape}")

print("\n--- Pitching Training Data Summary ---")
print("DataFrame head after data preparation:")
print(pitching_df_train.head())
print(f"DataFrame shape: {pitching_df_train.shape}")

# 4. Train/test split and train models for each stat group using training data
trained_hitting_models = {}
hitting_stats_to_predict = hitting_stats # All hitting_stats will be predicted for batters

print("\n--- Training Hitting Models ---")
if not hitting_df_train.empty:
    for stat in hitting_stats_to_predict:
        weighted_cols = [f'{s}_weighted' for s in hitting_stats] # Features are all hitting weighted stats
        # Filter df to only include rows where all weighted columns are present and not NaN/Inf
        df_cleaned = hitting_df_train.dropna(subset=weighted_cols + [f'target_{stat}']).replace([float('inf'), float('-inf')], float('nan')).dropna(subset=weighted_cols + [f'target_{stat}'])

        if df_cleaned.empty:
            print(f"Skipping training for Hitting {stat.upper()}: No valid data after cleaning.")
            continue

        X = df_cleaned[weighted_cols]
        y = df_cleaned[f'target_{stat}']

        if len(X) < 2:
            print(f"Skipping training for Hitting {stat.upper()}: Not enough data ({len(X)} samples) for train-test split.")
            continue

        test_size = 0.3 if len(X) * 0.3 >= 1 else (1 if len(X) > 1 else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        trained_hitting_models[stat] = model

        y_pred = model.predict(X_test)
        print(f'--- Hitting {stat.upper()} ---')
        print('MSE:', mean_squared_error(y_test, y_pred))
else:
    print("No hitting data available for training.")

trained_pitching_models = {}
pitching_stats_to_predict = pitching_stats # Only pitcher_strikeouts for now

print("\n--- Training Pitching Models ---")
if not pitching_df_train.empty:
     for stat in pitching_stats_to_predict:
        # Features for pitching model could be based on pitching weighted stats
        # For now, let's use only the pitcher_strikeouts_weighted feature
        weighted_cols = [f'{s}_weighted' for s in pitching_stats] # Features are pitcher weighted stats
        df_cleaned = pitching_df_train.dropna(subset=weighted_cols + [f'target_{stat}']).replace([float('inf'), float('-inf')], float('nan')).dropna(subset=weighted_cols + [f'target_{stat}'])

        if df_cleaned.empty:
            print(f"Skipping training for Pitching {stat.upper()}: No valid data after cleaning.")
            continue

        # Ensure the features are in a list of lists or DataFrame for the model
        X = df_cleaned[weighted_cols]
        y = df_cleaned[f'target_{stat}']

        if len(X) < 2:
            print(f"Skipping training for Pitching {stat.upper()}: Not enough data ({len(X)} samples) for train-test split.")
            continue

        test_size = 0.3 if len(X) * 0.3 >= 1 else (1 if len(X) > 1 else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        trained_pitching_models[stat] = model

        y_pred = model.predict(X_test)
        print(f'--- Pitching {stat.upper()} ---')
        print('MSE:', mean_squared_error(y_test, y_pred))
else:
    print("No pitching data available for training.")

# 5. Create a prediction function
def predict_player_stats(player_id, trained_hitting_models, trained_pitching_models, api_hitting_stat_map, api_pitching_stat_map, hitting_stats, pitching_stats, player_info_map):
    """Predicts the next game stats for a given player ID based on their position."""
    player_info = player_info_map.get(player_id)
    if not player_info:
        # print(f"Player ID {player_id} not found in fetched data.")
        return None

    player_name = player_info.get('fullName', f"ID: {player_id}")
    primary_position_code = player_info.get('primaryPosition', {}).get('code', 'Unknown')
    is_pitcher = primary_position_code in ['P', 'SP', 'RP']

    # Only attempt prediction for players with known positions
    if primary_position_code == 'Unknown':
         # print(f"Skipping prediction for player {player_name}: Unknown position.")
         return None

    print(f"\nGenerating prediction for {player_name} (ID: {player_id}, Pos: {primary_position_code})")

    try:
        all_logs = []
        # Determine which game logs to fetch and which models/stats to use for prediction
        if is_pitcher:
            log_group = 'pitching'
            stats_to_process = pitching_stats
            api_stat_map_current = api_pitching_stat_map
            trained_models = trained_pitching_models
            weighted_cols = [f'{s}_weighted' for s in pitching_stats]
            print(f"  -> Classified as Pitcher for prediction. Fetching {log_group} logs.")
        else:
            log_group = 'hitting'
            stats_to_process = hitting_stats
            api_stat_map_current = api_hitting_stat_map
            trained_models = trained_hitting_models
            weighted_cols = [f'{s}_weighted' for s in hitting_stats]
            print(f"  -> Classified as Hitter for prediction. Fetching {log_group} logs.")

        if not stats_to_process:
             print(f"No relevant stats defined for position {primary_position_code}. Skipping prediction.")
             return None

        # Fetch game logs for the training years (2023, 2024) AND the prediction year (2025)
        prediction_years = [year_minus_1, year_minus_2, year_plus_0]

        for year in prediction_years:
            game_log_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&season={year}&group={log_group}"
            response = requests.get(game_log_url)
            response.raise_for_status()
            player_stats_data = response.json()

            player_logs = []
            if player_stats_data and 'stats' in player_stats_data and len(player_stats_data['stats']) > 0 and 'splits' in player_stats_data['stats'][0]:
                 player_logs = player_stats_data['stats'][0]['splits']

            if player_logs:
                 game_data_list = []
                 for game in player_logs:
                      game_date = game.get('date')
                      game_stats = game.get('stat', {}) # Get the nested stat dictionary
                      game_data = {'game_date': game_date}
                      for stat_name, api_key in api_stat_map_current.items():
                           game_data[api_key] = game_stats.get(api_key, 0)
                      if all(api_key in game_stats for api_key in api_stat_map_current.values()):
                           game_data_list.append(game_data)
                 if game_data_list:
                      all_logs.append(pd.DataFrame(game_data_list))

        if not all_logs:
            print(f"No valid {log_group} game logs found for prediction for player {player_id} across {year_minus_2}, {year_minus_1}, and {year_plus_0}")
            return None

        logs = pd.concat(all_logs).dropna(subset=['game_date']) # Concatenate and drop rows without date

        # Need at least one game to calculate features for prediction
        if logs.empty or len(logs) < 1:
             print(f"Not enough {log_group} game logs ({len(logs)} found) for prediction for player {player_id} across {year_minus_2}, {year_minus_1}, and {year_plus_0}")
             return None

        logs['game_date'] = pd.to_datetime(logs['game_date'])
        logs = logs.sort_values('game_date').reset_index(drop=True)

        # Features for prediction are based on ALL available logs (2023, 2024, 2025)
        historical_logs = logs.copy() # Use all logs as historical for prediction features

        # Calculate last 10 games from the most recent season in historical logs for prediction
        latest_year_in_logs = historical_logs['game_date'].dt.year.max()
        if latest_year_in_logs:
            last_10_logs = historical_logs[historical_logs['game_date'].dt.year == latest_year_in_logs].tail(10)
        else:
             last_10_logs = pd.DataFrame() # Empty if no latest year found

        features = {}
        for stat_name, api_key in api_stat_map_current.items():
             last_10_avg = last_10_logs[api_key].mean() if api_key in last_10_logs.columns and not last_10_logs.empty else 0
             hist_avg = historical_logs[api_key].mean() if api_key in historical_logs.columns and not historical_logs.empty else 0
             features[f"{stat_name}_weighted"] = 0.75 * last_10_avg + 0.25 * hist_avg # Use 75/25 weighting for prediction

        # Create a DataFrame for prediction
        X_predict = pd.DataFrame([features])
        if not trained_models:
             print(f"No models were trained for {log_group}. Cannot make predictions.")
             return None

        X_predict = X_predict.reindex(columns=weighted_cols, fill_value=0)

        predictions = {}
        for stat in stats_to_process:
            if stat in trained_models: # Check if a model was trained for this stat
                model = trained_models[stat]
                prediction = model.predict(X_predict)[0]
                predictions[stat] = max(0, round(prediction)) # Predictions can't be negative
            else:
                print(f"No model trained for {stat}. Skipping prediction.")
                predictions[stat] = None

        return predictions

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {log_group} game logs for prediction for player {player_id}: {e}")
        return None
    except Exception as e:
        print(f"Error processing data for prediction for player {player_id}: {e}")
        return None

# 6. Train models for each stat group and store them
# Models are trained on 2023 and 2024 data (hitting_df_train, pitching_df_train)

# Optional: Loop through all players and print predictions using 2023, 2024, and 2025 data
print(f"\n--- Predictions for All Players (based on data through {year_plus_0})---")

# Re-fetch player IDs including those who may only appear in the current year (2025)
# This ensures we attempt predictions for players active in 2025 even if they weren't in the last two full seasons
all_player_ids = set()
all_player_info_map = {}

for year in [year_minus_1, year_minus_2, year_plus_0]:
    for group in ['hitting', 'pitching']:
        season_stats_url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&group={group}&season={year}"
        try:
            response = requests.get(season_stats_url)
            response.raise_for_status()
            season_stats_data = response.json()
            if season_stats_data and 'stats' in season_stats_data and len(season_stats_data['stats']) > 0 and 'splits' in season_stats_data['stats'][0]:
                player_splits = season_stats_data['stats'][0]['splits']
                for entry in player_splits:
                    player_info = entry.get('player')
                    if player_info and 'id' in player_info:
                        player_id = player_info['id']
                        all_player_ids.add(player_id)
                        # Update info, prioritizing later years/pitching position if found
                        current_info = all_player_info_map.get(player_id, {})
                        new_info = {
                            'fullName': player_info.get('fullName', f'Player ID: {player_id}'),
                             'primaryPosition': player_info.get('primaryPosition', current_info.get('primaryPosition', {}))
                        }
                        # If the new position is pitching and old wasn't, or no old position, update
                        if new_info['primaryPosition'].get('code') in ['P', 'SP', 'RP'] and current_info.get('primaryPosition', {}).get('code') not in ['P', 'SP', 'RP']:
                            all_player_info_map[player_id] = new_info
                        elif player_id not in all_player_info_map:
                            all_player_info_map[player_id] = new_info
                        # Also update name if more recent data has it (less critical)
                        elif 'fullName' in player_info:
                             all_player_info_map[player_id]['fullName'] = player_info['fullName']

        except requests.exceptions.RequestException as e:
            print(f"Error fetching season stats for player list in {year} ({group}): {e}")
        time.sleep(0.05) # Small delay

# Attempt to enrich player info with position for all found players if missing
print("Attempting to enrich position information for all players found...")
final_player_info_map = {}
for player_id in all_player_ids:
    if player_id not in all_player_info_map or not all_player_info_map[player_id].get('primaryPosition', {}).get('code'):
        details = get_player_details(player_id)
        if details:
             final_player_info_map[player_id] = details
        else:
             final_player_info_map[player_id] = all_player_info_map.get(player_id, {'fullName': f'Player ID: {player_id}', 'primaryPosition': {}})
    else:
        final_player_info_map[player_id] = all_player_info_map[player_id]
    time.sleep(0.05) # Small delay

# Use the comprehensive list of players for prediction
player_ids_for_prediction = list(final_player_info_map.keys())
player_info_map_for_prediction = final_player_info_map

print(f"Attempting to predict for {len(player_ids_for_prediction)} players active in {year_minus_2}, {year_minus_1}, or {year_plus_0}.")

for player_id in player_ids_for_prediction:
    predicted_stats = predict_player_stats(player_id, trained_hitting_models, trained_pitching_models, api_hitting_stat_map, api_pitching_stat_map, hitting_stats, pitching_stats, player_info_map_for_prediction)
    if predicted_stats:
        player_info = player_info_map_for_prediction.get(player_id, {})
        player_name = player_info.get('fullName', f"ID: {player_id}")
        primary_position_code = player_info.get('primaryPosition', {}).get('code', 'Unknown')
        is_pitcher = primary_position_code in ['P', 'SP', 'RP']

        # Determine which stats to display based on position
        stats_to_display = pitching_stats if is_pitcher else hitting_stats

        print(f"\n{player_name} (ID: {player_id}, Pos: {primary_position_code}):")
        for stat in stats_to_display:
            if stat in predicted_stats and predicted_stats[stat] is not None:
                # Format stat name for output, specifically for strikeouts
                if stat == 'strikeouts':
                    formatted_stat_name = "Pitcher Strikeouts" if is_pitcher else "Hitter Strikeouts"
                else:
                     formatted_stat_name = stat.replace('_weighted', '').replace('total_bases', 'Total Bases').replace('home_runs', 'Home Runs').replace('rbi', 'RBI').capitalize()
                print(f"  {formatted_stat_name}: {predicted_stats[stat]}")
            elif stat in predicted_stats:
                 # Print stat name even if prediction is None (e.g., model not trained)
                 if stat == 'strikeouts':
                    formatted_stat_name = "Pitcher Strikeouts" if is_pitcher else "Hitter Strikeouts"
                 else:
                     formatted_stat_name = stat.replace('_weighted', '').replace('total_bases', 'Total Bases').replace('home_runs', 'Home Runs').replace('rbi', 'RBI').capitalize()
                 print(f"  {formatted_stat_name}: N/A (Model not trained or data issue)")