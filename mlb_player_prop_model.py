#pip install pybaseball unidecode thefuzz
import pandas as pd
import pybaseball
from pybaseball import batting_stats, statcast_batter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
import time # Import time for rate limiting
import datetime # Import datetime to get current year
import numpy as np # Import numpy for nanmean and average
import subprocess # Import subprocess to run the scraper
import os # Import os for file path manipulation
import logging # Import logging
from unidecode import unidecode # Import unidecode for name normalization
from thefuzz import process # Import thefuzz for fuzzy matching

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping of team abbreviations from scraped data to full team names from MLB API
TEAM_ABBREVIATION_MAP = {
    "ARI": "Arizona Diamondbacks", "AZ": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox", "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres", "SDP": "San Diego Padres",
    "SF": "San Francisco Giants", "SFG": "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals", "WSN": "Washington Nationals",
}

# Get current year dynamically
current_year = datetime.datetime.now().year
year_minus_1 = current_year - 1
year_minus_2 = current_year - 2
year_plus_0 = current_year # This will be 2025 if run in 2025

# Function to get player details including primary position and team
def get_player_details(player_id):
    details_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=currentTeam"
    try:
        response = requests.get(details_url)
        response.raise_for_status()
        details_data = response.json()
        if details_data and 'people' in details_data and len(details_data['people']) > 0:
            person_data = details_data['people'][0]
            team_info = person_data.get('currentTeam', {})
            return {
                'fullName': person_data.get('fullName', f'Player ID: {player_id}'),
                'primaryPosition': person_data.get('primaryPosition', {}),
                'teamName': team_info.get('name'),
                'teamId': team_info.get('id')
            }
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching details for player {player_id}: {e}")
    return None

# 1. Get player list, names, and positions for the last two years (using MLB Stats API season stats)
player_info_map = {} # Dictionary to store player info (name, position, team)
player_ids = set() # Use a set to store unique player IDs

# Fetch initial player list from the last two *full* seasons for training data identification
training_years = [year_minus_1, year_minus_2, year_plus_0] # Include current year for player info

try:
    # Fetch Hitting Season Stats for the last two full years
    for year in training_years:
        season_stats_url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season={year}"
        logging.info(f"Fetching hitting season stats for {year} from: {season_stats_url}")
        response = requests.get(season_stats_url)
        response.raise_for_status()
        season_stats_data = response.json()

        if season_stats_data and 'stats' in season_stats_data and len(season_stats_data['stats']) > 0 and 'splits' in season_stats_data['stats'][0]:
            player_splits = season_stats_data['stats'][0]['splits']
            logging.info(f"Found {len(player_splits)} hitting player entries (splits) in {year} season stats.")
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
        logging.info(f"Fetching pitching season stats for {year} from: {season_stats_url}")
        response = requests.get(season_stats_url)
        response.raise_for_status()
        season_stats_data = response.json()

        if season_stats_data and 'stats' in season_stats_data and len(season_stats_data['stats']) > 0 and 'splits' in season_stats_data['stats'][0]:
            player_splits = season_stats_data['stats'][0]['splits']
            logging.info(f"Found {len(player_splits)} pitching player entries (splits) in {year} season stats.")
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
    logging.error(f"Error fetching season stats: {e}")
    player_ids = set() # Reset to empty if fetching fails
    player_info_map = {} # Reset mapping if fetching fails

player_ids = list(player_ids) # Convert set to list for iteration
logging.info(f"Found {len(player_ids)} unique player IDs from {', '.join(map(str, training_years))} season stats for training.")

# Attempt to enrich player info with position and team information...
logging.info("Attempting to enrich player position and team information...")
updated_player_info_map = {}
for player_id in player_ids:
    # Get details for every player to ensure team info is always present
    details = get_player_details(player_id)
    if details:
         updated_player_info_map[player_id] = details
    elif player_id in player_info_map:
         # If details fetch fails, fall back to existing info from season stats
         updated_player_info_map[player_id] = player_info_map[player_id]
    else:
         # If all else fails, create a minimal entry
         updated_player_info_map[player_id] = {'fullName': f'Player ID: {player_id}', 'primaryPosition': {}}
    time.sleep(0.05) # Small delay for details API calls

player_info_map = updated_player_info_map # Update the main map

# 2. Define stats relevant to hitting and pitching
hitting_stats = ['runs', 'total_bases', 'hits', 'strikeouts'] # Strikeouts as batter strikeouts
pitching_stats = ['strikeouts', 'earned_runs_allowed'] # Strikeouts as pitcher strikeouts, and Earned Runs Allowed

# Mapping from our stat names to potential API field names
api_hitting_stat_map = {
    'runs': 'runs',
    'total_bases': 'totalBases',
    'hits': 'hits',
    'strikeouts': 'strikeOuts', # API key for batter strikeouts
}

api_pitching_stat_map = {
    'strikeouts': 'strikeOuts', # API key for pitcher strikeouts
    'earned_runs_allowed': 'earnedRuns', # API key for earned runs allowed
}

# 3. Prepare data for TRAINING (using MLB Stats API game logs for 2023 and 2024)
# This will now build separate data for batters and pitchers for training
hitting_data_train = []
pitching_data_train = []

training_years_for_logs = [year_minus_1, year_minus_2] # Use only these two years for training logs

logging.info(f"--- Preparing Training Data ({year_minus_2}-{year_minus_1}) ---")

# Use the comprehensive player map to get player IDs for training data (only include those with position info)
training_player_ids = [pid for pid, info in player_info_map.items() if info.get('primaryPosition', {}).get('type') != 'Unknown']

logging.info(f"Preparing training data for {len(training_player_ids)} players with known positions from {year_minus_2}-{year_minus_1}.")

for player_id in training_player_ids:
    player_info = player_info_map.get(player_id, {})
    player_name_for_debug = player_info.get('fullName', f'ID: {player_id}')
    primary_position_type = player_info.get('primaryPosition', {}).get('type', 'Unknown') # Changed to 'type'
    is_pitcher = primary_position_type == 'Pitcher' # Changed to check 'Pitcher' type

    logging.info(f"Preparing training data for {player_name_for_debug} (ID: {player_id}, Pos: {primary_position_type})...") # Changed to 'type'

    try:
        all_logs = []
        # Determine which game logs to fetch based on primary position for training
        if is_pitcher:
            log_group = 'pitching'
            stats_to_process = pitching_stats
            api_stat_map_current = api_pitching_stat_map
            logging.info(f"  -> Classified as Pitcher. Fetching pitching logs for training.")
        else:
            log_group = 'hitting'
            stats_to_process = hitting_stats
            api_stat_map_current = api_hitting_stat_map
            logging.info(f"  -> Classified as Hitter. Fetching hitting logs for training.")

        if not stats_to_process: # Skip if no relevant stats for this position
             logging.info(f"  No relevant stats defined for position {primary_position_type}. Skipping training data for {player_name_for_debug}.") # Changed to 'type'
             time.sleep(0.05)
             continue

        # Fetch game logs for the training years
        for year in training_years_for_logs: # Now explicitly using training_years_for_logs
            game_log_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&season={year}&group={log_group}"
            logging.info(f"  Fetching game logs for {player_name_for_debug} ({log_group}) for year {year} from: {game_log_url}")
            response = requests.get(game_log_url)
            response.raise_for_status()
            player_stats_data = response.json()
            logging.info(f"  Raw API response for {player_name_for_debug} ({log_group}) for year {year}: {player_stats_data}") # New debug print

            player_logs = []
            if player_stats_data and 'stats' in player_stats_data and len(player_stats_data['stats']) > 0 and 'splits' in player_stats_data['stats'][0]:
                 player_logs = player_stats_data['stats'][0]['splits']
                 logging.info(f"  Raw {log_group} player logs for {player_name_for_debug} in {year}: {player_logs[:2]}... (showing first 2 entries)" if player_logs else f"  No raw {log_group} logs found for {player_name_for_debug} in {year}")

            if player_logs:
                 logging.info(f"  Found {len(player_logs)} {log_group} game logs for {year}.")
                 game_data_list = []
                 for game in player_logs:
                      game_date = game.get('date')
                      game_stats = game.get('stat', {}) # Get the nested stat dictionary
                      logging.debug(f"    Processing game date {game_date}, raw stats: {game_stats}") # Use debug for verbose raw stats
                      game_data = {'game_date': game_date}
                      all_api_keys_present = True
                      for stat_name, api_key in api_stat_map_current.items():
                          val = game_stats.get(api_key)
                          if val is not None: # Ensure the stat value is actually present
                             game_data[api_key] = val
                          else:
                             all_api_keys_present = False
                             logging.info(f"    Missing API key '{api_key}' for {stat_name} in game {game_date} for {player_name_for_debug}. Skipping this game.")
                             break # A required API key is missing for this game
                      if all_api_keys_present:
                           game_data_list.append(game_data)
                 if game_data_list:
                      logging.info(f"  {len(game_data_list)} valid game data entries after processing stats for {player_name_for_debug} in {year}.")
                      all_logs.append(pd.DataFrame(game_data_list))
                 else:
                      logging.info(f"  No valid game data entries found for {player_name_for_debug} in {year} after checking for all API keys.")

        if not all_logs:
            logging.info(f"  No valid {log_group} game logs found for training for {player_name_for_debug} across {', '.join(map(str, training_years_for_logs))}. Skipping.") # Changed to training_years_for_logs
            time.sleep(0.05)
            continue

        logs = pd.concat(all_logs).dropna(subset=['game_date']) # Concatenate and drop rows without date
        logging.info(f"  Concatenated {log_group} logs for {player_name_for_debug}, shape: {logs.shape}")

        # Require at least two games to calculate features and have a target game for training
        if logs.empty or len(logs) < 2:
            logging.info(f"  Not enough {log_group} game logs ({len(logs)} found) for training for {player_name_for_debug}. Skipping.")
            time.sleep(0.05)
            continue

        logs['game_date'] = pd.to_datetime(logs['game_date'])
        logs = logs.sort_values('game_date').reset_index(drop=True)

        # Features for training are based on all but the last game in the logs
        historical_logs = logs.iloc[:-1]

        # Ensure there's at least one game left after slicing for historical_logs
        if historical_logs.empty:
            logging.info(f"  Not enough historical {log_group} game logs for {player_name_for_debug} after taking last game as target. Skipping.")
            continue

        # Calculate various historical features
        features = {}
        for stat_name, api_key in api_stat_map_current.items():
             # Ensure stat column exists before calculating
             if api_key in historical_logs.columns:
                 # Rolling 10-game average from the most recent season in historical logs
                 latest_year_in_logs = historical_logs['game_date'].dt.year.max()
                 if latest_year_in_logs:
                     # Filter for the latest year and then apply rolling mean
                     latest_year_data = historical_logs[historical_logs['game_date'].dt.year == latest_year_in_logs][api_key]
                     rolling_10_logs = latest_year_data.rolling(window=10).mean().iloc[-1] if len(latest_year_data) >= 10 else np.nan
                 else:
                     rolling_10_logs = np.nan

                 # Rolling 5-game average (based on the last game in historical_logs)
                 rolling_5_avg = historical_logs[api_key].rolling(window=5).mean().iloc[-1] if len(historical_logs) >= 5 else np.nan

                 # Rolling 3-game average (based on the last game in historical_logs)
                 rolling_3_avg = historical_logs[api_key].rolling(window=3).mean().iloc[-1] if len(historical_logs) >= 3 else np.nan

                 # Overall historical average (based on all historical logs)
                 hist_avg = historical_logs[api_key].mean() if not historical_logs.empty else 0

                 # Combine recent averages using specified weights, ignoring NaN values
                 recent_averages_with_weights = []
                 if not np.isnan(rolling_10_logs):
                      recent_averages_with_weights.append((rolling_10_logs, 0.25))
                 if not np.isnan(rolling_5_avg):
                      recent_averages_with_weights.append((rolling_5_avg, 0.35))
                 if not np.isnan(rolling_3_avg):
                      recent_averages_with_weights.append((rolling_3_avg, 0.4))

                 if recent_averages_with_weights:
                      averages = [item[0] for item in recent_averages_with_weights]
                      weights = [item[1] for item in recent_averages_with_weights]
                      # Normalize weights if not all components are present
                      normalized_weights = np.array(weights) / np.sum(weights)
                      recent_weighted_avg = np.average(averages, weights=normalized_weights)
                 else:
                      recent_weighted_avg = 0

                 # Create the final weighted feature (75% recent, 25% historical)
                 features[f"{stat_name}_weighted_feature"] = 0.75 * recent_weighted_avg + 0.25 * hist_avg

             else:
                 # Add the combined weighted feature with 0 if stat column doesn't exist
                 features[f"{stat_name}_weighted_feature"] = 0

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
        logging.error(f"Error fetching {log_group} game logs for training for player {player_name_for_debug}: {e}")
    except Exception as e:
        logging.error(f"Error processing training data for player {player_name_for_debug}: {e}")

    time.sleep(0.05) # Add a small delay to avoid overwhelming the API

# Create separate DataFrames for hitting and pitching training data
hitting_df_train = pd.DataFrame(hitting_data_train)
pitching_df_train = pd.DataFrame(pitching_data_train)

logging.info(f"--- Hitting Training Data Summary ---")
logging.info("DataFrame head after data preparation:")
logging.info(f"{hitting_df_train.head()}") # Use f-string for DataFrame display
logging.info(f"DataFrame shape: {hitting_df_train.shape}")

logging.info("--- Pitching Training Data Summary ---")
logging.info("DataFrame head after data preparation:")
logging.info(f"{pitching_df_train.head()}") # Use f-string for DataFrame display
logging.info(f"DataFrame shape: {pitching_df_train.shape}")

# 4. Train/test split and train models for each stat group using training data
trained_hitting_models = {}
hitting_stats_to_predict = hitting_stats # All hitting_stats will be predicted for batters

# Define the single weighted hitting feature column name
hitting_weighted_feature_cols = [f'{stat}_weighted_feature' for stat in hitting_stats]

logging.info("--- Training Hitting Models ---")
if not hitting_df_train.empty:
    for stat in hitting_stats_to_predict:
        # The feature for this stat is the single weighted feature
        feature_col = f'{stat}_weighted_feature'
        target_col = f'target_{stat}'

        # Filter df to only include rows where the feature and target columns are present and not NaN/Inf
        cols_to_check = [feature_col, target_col]

        # Ensure all necessary feature columns exist in the DataFrame before cleaning
        valid_cols_to_check = [col for col in cols_to_check if col in hitting_df_train.columns]

        if len(valid_cols_to_check) < len(cols_to_check):
             logging.info(f"Skipping training for Hitting {stat.upper()}: Missing one or more feature/target columns.")
             continue

        df_cleaned = hitting_df_train.dropna(subset=valid_cols_to_check).replace([float('inf'), float('-inf')], float('nan')).dropna(subset=valid_cols_to_check)

        if df_cleaned.empty:
            logging.info(f"Skipping training for Hitting {stat.upper()}: No valid data after cleaning.")
            continue

        # Use only the single weighted feature for training this model
        X = df_cleaned[[feature_col]] # X needs to be a DataFrame (list of columns)
        y = df_cleaned[target_col]

        if len(X) < 2:
            logging.info(f"Skipping training for Hitting {stat.upper()}: Not enough data ({len(X)} samples) for train-test split.")
            continue

        test_size = 0.3 if len(X) * 0.3 >= 1 else (1 if len(X) > 1 else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        trained_hitting_models[stat] = model

        y_pred = model.predict(X_test)
        logging.info(f'--- Hitting {stat.upper()} ---')
        logging.info('MSE:', mean_squared_error(y_test, y_pred))
else:
    logging.info("No hitting data available for training.")

trained_pitching_models = {}
pitching_stats_to_predict = pitching_stats # Only pitcher_strikeouts for now

# Define the single weighted pitching feature column name
pitching_weighted_feature_cols = [f'{stat}_weighted_feature' for stat in pitching_stats]

logging.info("--- Training Pitching Models ---")
if not pitching_df_train.empty:
     for stat in pitching_stats_to_predict:
        # The feature for this stat is the single weighted feature
        feature_col = f'{stat}_weighted_feature'
        target_col = f'target_{stat}'

        # Filter df to only include rows where the feature and target columns are present and not NaN/Inf
        cols_to_check = [feature_col, target_col]

        # Ensure all necessary feature columns exist in the DataFrame before cleaning
        valid_cols_to_check = [col for col in cols_to_check if col in pitching_df_train.columns]

        if len(valid_cols_to_check) < len(cols_to_check):
             logging.info(f"Skipping training for Pitching {stat.upper()}: Missing one or more feature/target columns.")
             continue

        df_cleaned = pitching_df_train.dropna(subset=valid_cols_to_check).replace([float('inf'), float('-inf')], float('nan')).dropna(subset=valid_cols_to_check)

        if df_cleaned.empty:
            logging.info(f"Skipping training for Pitching {stat.upper()}: No valid data after cleaning.")
            continue

        # Ensure the features are in a list of lists or DataFrame for the model
        X = df_cleaned[[feature_col]] # X needs to be a DataFrame (list of columns)
        y = df_cleaned[target_col]

        if len(X) < 2:
            logging.info(f"Skipping training for Pitching {stat.upper()}: Not enough data ({len(X)} samples) for train-test split.")
            continue

        test_size = 0.3 if len(X) * 0.3 >= 1 else (1 if len(X) > 1 else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        trained_pitching_models[stat] = model

        y_pred = model.predict(X_test)
        logging.info(f'--- Pitching {stat.upper()} ---')
        logging.info('MSE:', mean_squared_error(y_test, y_pred))
else:
    logging.info("No pitching data available for training.")

# 5. Create a prediction function
def predict_player_stats(player_id, trained_hitting_models, trained_pitching_models, api_hitting_stat_map, api_pitching_stat_map, hitting_stats, pitching_stats, player_info_map):
    """Predicts the next game stats for a given player ID based on their position."""
    player_info = player_info_map.get(player_id)
    if not player_info:
        logging.info(f"Player ID {player_id} not found in fetched data for prediction.")
        return None

    player_name = player_info.get('fullName', f"ID: {player_id}")
    primary_position_type = player_info.get('primaryPosition', {}).get('type', 'Unknown') # Changed to 'type'
    is_pitcher = primary_position_type == 'Pitcher' # Changed to check 'Pitcher' type

    # Only attempt prediction for players with known positions
    if primary_position_type == 'Unknown': # Changed to 'type'
         logging.info(f"Skipping prediction for player {player_name}: Unknown position.")
         return None

    logging.info(f"Generating prediction for {player_name} (ID: {player_id}, Pos: {primary_position_type})") 

    try:
        all_logs = []
        # Determine which game logs to fetch and which models/stats to use for prediction
        if is_pitcher:
            log_group = 'pitching'
            stats_to_process = pitching_stats
            api_stat_map_current = api_pitching_stat_map
            trained_models = trained_pitching_models
            # The feature for this stat is the single weighted feature
            feature_cols = [f'{stat}_weighted_feature' for stat in stats_to_process]
            logging.info(f"  -> Classified as Pitcher for prediction. Fetching {log_group} logs.") # Changed log message
        else:
            log_group = 'hitting'
            stats_to_process = hitting_stats
            api_stat_map_current = api_hitting_stat_map
            trained_models = trained_hitting_models
            # The feature for this stat is the single weighted feature
            feature_cols = [f'{stat}_weighted_feature' for stat in stats_to_process]
            logging.info(f"  -> Classified as Hitter for prediction. Fetching {log_group} logs.") # Changed log message

        if not stats_to_process:
             logging.info(f"No relevant stats defined for position {primary_position_type}. Skipping prediction for {player_name}.") # Changed to 'type'
             return None

        # Fetch game logs for the training years (2023, 2024) AND the prediction year (2025)
        prediction_years = [year_minus_1, year_minus_2, year_plus_0]

        for year in prediction_years:
            game_log_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&season={year}&group={log_group}"
            logging.info(f"  Fetching game logs for {player_name} ({log_group}) for year {year} from: {game_log_url}")
            response = requests.get(game_log_url)
            response.raise_for_status()
            player_stats_data = response.json()
            logging.info(f"  Raw API response for {player_name} ({log_group}) for year {year}: {player_stats_data}") # New debug print

            player_logs = []
            if player_stats_data and 'stats' in player_stats_data and len(player_stats_data['stats']) > 0 and 'splits' in player_stats_data['stats'][0]:
                 player_logs = player_stats_data['stats'][0]['splits']
                 logging.info(f"  Raw {log_group} player logs for {player_name} in {year}: {player_logs[:2]}... (showing first 2 entries)" if player_logs else f"  No raw {log_group} logs found for {player_name} in {year}")

            if player_logs:
                 logging.info(f"  Found {len(player_logs)} {log_group} game logs for {year}.")
                 game_data_list = []
                 for game in player_logs:
                      game_date = game.get('date')
                      game_stats = game.get('stat', {}) # Get the nested stat dictionary
                      logging.debug(f"    Processing game date {game_date}, raw stats: {game_stats}") # Use debug for verbose raw stats
                      game_data = {'game_date': game_date}
                      all_api_keys_present = True
                      for stat_name, api_key in api_stat_map_current.items():
                          val = game_stats.get(api_key)
                          if val is not None: # Ensure the stat value is actually present
                             game_data[api_key] = val
                          else:
                             all_api_keys_present = False
                             logging.info(f"    Missing API key '{api_key}' for {stat_name} in game {game_date} for {player_name}. Skipping this game.")
                             break # A required API key is missing for this game
                      if all_api_keys_present:
                           game_data_list.append(game_data)
                 if game_data_list:
                      logging.info(f"  {len(game_data_list)} valid game data entries after processing stats for {player_name} in {year}.")
                      all_logs.append(pd.DataFrame(game_data_list))
                 else:
                      logging.info(f"  No valid game data entries found for {player_name} in {year} after checking for all API keys.")

        if not all_logs:
            logging.info(f"No valid {log_group} game logs found for prediction for player {player_name} across {', '.join(map(str, prediction_years))}")
            # If no logs are found at all, predict 0 for all relevant stats
            return {stat: 0 for stat in stats_to_process}

        logs = pd.concat(all_logs).dropna(subset=['game_date']) # Concatenate and drop rows without date
        logging.info(f"  Concatenated {log_group} logs for {player_name}, shape: {logs.shape}")

        # Need at least one game to calculate features for prediction
        if logs.empty or len(logs) < 1:
             logging.info(f"Not enough {log_group} game logs ({len(logs)} found) for prediction for player {player_name} across {', '.join(map(str, prediction_years))}")
             # If not enough logs, predict 0 for all relevant stats
             return {stat: 0 for stat in stats_to_process}

        logs['game_date'] = pd.to_datetime(logs['game_date'])
        logs = logs.sort_values('game_date').reset_index(drop=True)

        # Features for prediction are based on ALL available logs (2023, 2024, 2025)
        historical_logs = logs.copy() # Use all logs as historical for prediction features

        # Calculate various historical features for prediction
        features = {}
        for stat_name, api_key in api_stat_map_current.items():
            if api_key in historical_logs.columns:
                # Rolling 10-game average from the most recent season in historical logs
                latest_year_in_logs = historical_logs['game_date'].dt.year.max()
                if latest_year_in_logs:
                    # Filter for the latest year and then apply rolling mean
                    latest_year_data = historical_logs[historical_logs['game_date'].dt.year == latest_year_in_logs][api_key]
                    rolling_10_logs = latest_year_data.rolling(window=10).mean().iloc[-1] if len(latest_year_data) >= 10 else np.nan
                else:
                    rolling_10_logs = np.nan

                # Rolling 5-game average (based on the last game in historical_logs)
                rolling_5_avg = historical_logs[api_key].rolling(window=5).mean().iloc[-1] if len(historical_logs) >= 5 else np.nan

                # Rolling 3-game average (based on the last game in historical_logs)
                rolling_3_avg = historical_logs[api_key].rolling(window=3).mean().iloc[-1] if len(historical_logs) >= 3 else np.nan

                # Overall historical average (based on all historical logs)
                hist_avg = historical_logs[api_key].mean() if not historical_logs.empty else 0

                # Combine recent averages using specified weights, ignoring NaN values
                recent_averages_with_weights = []
                if not np.isnan(rolling_10_logs):
                     recent_averages_with_weights.append((rolling_10_logs, 0.25))
                if not np.isnan(rolling_5_avg):
                     recent_averages_with_weights.append((rolling_5_avg, 0.35))
                if not np.isnan(rolling_3_avg):
                     recent_averages_with_weights.append((rolling_3_avg, 0.4))

                if recent_averages_with_weights:
                      averages = [item[0] for item in recent_averages_with_weights]
                      weights = [item[1] for item in recent_averages_with_weights]
                      # Normalize weights if not all components are present
                      normalized_weights = np.array(weights) / np.sum(weights)
                      recent_weighted_avg = np.average(averages, weights=normalized_weights)
                else:
                      recent_weighted_avg = 0

                # Create the final weighted feature (75% recent, 25% historical)
                features[f"{stat_name}_weighted_feature"] = 0.75 * recent_weighted_avg + 0.25 * hist_avg

            else:
                 # Add the combined weighted feature with 0 if stat column doesn't exist
                 features[f"{stat_name}_weighted_feature"] = 0

        # Create a DataFrame for prediction
        X_predict = pd.DataFrame([features])
        if not trained_models:
             logging.info(f"No models were trained for {log_group}. Cannot make predictions.")
             return None

        # Ensure the prediction DataFrame has all the features the model was trained on
        # If a player is missing a certain historical average (e.g., not enough games), fill with 0
        # Use the correct feature columns list based on position
        X_predict = X_predict.reindex(columns=feature_cols, fill_value=0)

        predictions = {}
        for stat in stats_to_process:
            if stat in trained_models: # Check if a model was trained for this stat
                model = trained_models[stat]
                # Predict using the single weighted feature for this stat
                prediction = model.predict(X_predict[[f'{stat}_weighted_feature']])[0]
                predictions[stat] = max(0, round(prediction)) # Predictions can't be negative
            else:
                logging.info(f"No model trained for {stat}. Skipping prediction.")
                predictions[stat] = 0 # Default to 0 if model not found for this stat

        # Post-prediction adjustments for logical consistency (Batting Stats):
        if log_group == 'hitting':
            predicted_hits = predictions.get('hits', 0)
            predicted_total_bases = predictions.get('total_bases', 0)

            # Rule: If hits == 0 ⇒ total_bases must also be 0
            if predicted_hits == 0:
                if predicted_total_bases > 0:
                    logging.info(f"Adjusting predicted Total Bases for player {player_id} from {predicted_total_bases} to 0 (Hits is 0) for logical consistency.")
                    predictions['total_bases'] = 0
                    predicted_total_bases = 0 # Update for subsequent checks

            # Rule: Total bases ≥ hits (General rule, applied after others might have adjusted hits/total_bases)
            if predicted_total_bases < predicted_hits:
                logging.info(f"Adjusting predicted Total Bases for player {player_id} from {predicted_total_bases} to {predicted_hits} (at least Hits) for logical consistency.")
                predictions['total_bases'] = predicted_hits
                predicted_total_bases = predicted_hits # Update for subsequent checks

            # Rule: Max bases from hits is 4 per hit (Total Bases <= 4 * Hits)
            max_possible_bases_from_hits = predicted_hits * 4
            if predicted_total_bases > max_possible_bases_from_hits:
                 logging.info(f"Adjusting predicted Total Bases for player {player_id} from {predicted_total_bases} to {max_possible_bases_from_hits} (max 4 bases per hit) for logical consistency.")
                 predictions['total_bases'] = max_possible_bases_from_hits
                 predicted_total_bases = max_possible_bases_from_hits # Update for subsequent checks

        return "Success", predictions

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {log_group} game logs for prediction for player {player_name}: {e}")
        return "Prediction Error", None
    except Exception as e:
        logging.error(f"Error processing data for prediction for player {player_name}: {e}")
        return "Prediction Error", None

# Function to find player ID by name (now using team and fuzzy matching)
def find_player_id_by_name(player_name, team_abbreviation, player_info_map, team_map):
    """Finds the best player ID match using team and fuzzy name matching."""
    player_name_norm = unidecode(player_name).lower()
    target_team_name = team_map.get(team_abbreviation)
    
    possible_players = {}
    search_scope_message = ""
    team_search_failed = False

    if target_team_name:
        # Filter players by team first for a much more accurate search
        possible_players = {
            info['fullName']: pid
            for pid, info in player_info_map.items()
            if info.get('teamName') == target_team_name
        }
        search_scope_message = f"on team '{target_team_name}'"
        if not possible_players:
            team_search_failed = True
    else:
        team_search_failed = True

    # If no players were found for the team, or if the team was unknown, search all players
    if team_search_failed:
        log_message = f"Could not find players for team abbreviation '{team_abbreviation}'." if team_abbreviation else "No team provided."
        logging.warning(f"{log_message} Searching all players as a fallback.")
        possible_players = {info['fullName']: pid for pid, info in player_info_map.items() if info.get('fullName')}
        search_scope_message = "across all teams"

    if not possible_players:
        logging.error("Player info map is completely empty. Cannot find any player.")
        return None

    # Use fuzzy matching to find the best name match within the filtered list
    # A score_cutoff of 85 is a good balance between finding matches and avoiding mismatches
    best_match = process.extractOne(player_name_norm, possible_players.keys(), score_cutoff=85)

    if best_match:
        matched_name, score = best_match[0], best_match[1]
        player_id = possible_players[matched_name]
        logging.info(f"Fuzzy matched '{player_name}' with '{matched_name}' (Score: {score}) -> Player ID: {player_id}")
        return player_id
    else:
        logging.warning(f"Could not find a confident match for '{player_name}' {search_scope_message}.")
        return None

# Main execution block to run the scraper and then predict props
if __name__ == '__main__':
    # Define path relative to the script to ensure it finds the file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scraped_csv_path = os.path.join(script_dir, "scraped_prizepicks_props.csv")

    try:
        # Execute the scraper script (removed as per user request)
        # result = subprocess.run(
        #     ["python", prizepicks_scraper_path],
        #     capture_output=True, text=True, check=True
        # )
        # logging.info(result.stdout)
        # if result.stderr:
        #     logging.error("Scraper Errors:", result.stderr)

        # Directly read the CSV if it exists
        if os.path.exists(scraped_csv_path):
            scraped_props_df = pd.read_csv(scraped_csv_path)
            logging.info(f"Successfully loaded {len(scraped_props_df)} props from {scraped_csv_path}")
        else:
            logging.error(f"❌ Scraped data CSV not found at {scraped_csv_path}. Please run prizepicks_scraper.py first to generate it.")
            scraped_props_df = pd.DataFrame()

    except Exception as e:
        logging.error(f"An unexpected error occurred while reading the scraped CSV: {e}")
        scraped_props_df = pd.DataFrame()

    if scraped_props_df.empty:
        logging.info("No props to predict. Exiting.")
    else:
        logging.info("--- Training MLB Models ---")
        # Train models (existing training logic runs here automatically upon script execution)

        logging.info("--- Generating Player Prop Predictions ---")
        predicted_props_data = []

        # Get today's date for the output file
        game_date = datetime.date.today().strftime("%Y-%m-%d")

        # Mapping for output stat names to internal stat names
        prop_type_to_stat_name = {
            "Hits": "hits",
            "Total Bases": "total_bases",
            "Runs": "runs",
            "Pitcher Strikeouts": "strikeouts", # For pitchers
            "Hitter Strikeouts": "strikeouts", # For hitters
            "Earned Runs Allowed": "earned_runs_allowed"
        }

        for index, row in scraped_props_df.iterrows():
            player_name = row['player']
            prop_type = row['prop_type']
            original_line = row['line']
            original_team_field = row['team'] # e.g., 'BAL - C'
            original_opponent = row['opponent']
            
            predicted_value = 0.0 # Default to 0.0, ensures no NaN values
            prediction_status = "Unknown Error" # Default status

            # Attempt to parse team abbreviation
            team_abbreviation = original_team_field.split(' - ')[0].strip()

            # Skip combined player props
            if '+' in player_name:
                prediction_status = "Combined Player Prop"
            else:
                player_id = find_player_id_by_name(player_name, team_abbreviation, player_info_map, TEAM_ABBREVIATION_MAP)

                if player_id:
                    # Get all predicted stats for the player
                    status, predicted_stats = predict_player_stats(player_id, trained_hitting_models, trained_pitching_models, api_hitting_stat_map, api_pitching_stat_map, hitting_stats, pitching_stats, player_info_map)
                    prediction_status = status

                    if predicted_stats:
                        logging.info(f"  All Predicted Stats for {player_name}: {predicted_stats}")
                        if prop_type in prop_type_to_stat_name:
                            internal_stat_name = prop_type_to_stat_name[prop_type]
                            retrieved_value = predicted_stats.get(internal_stat_name)

                            # Special handling for strikeouts to ensure correct prop-position matching
                            if internal_stat_name == "strikeouts":
                                player_info = player_info_map.get(player_id, {})
                                primary_position_type = player_info.get('primaryPosition', {}).get('type', 'Unknown')
                                is_pitcher = primary_position_type == 'Pitcher'
                                
                                # Only assign value if the prop type matches the player's position
                                if (is_pitcher and prop_type == "Pitcher Strikeouts") or \
                                   (not is_pitcher and prop_type == "Hitter Strikeouts"):
                                    if retrieved_value is not None:
                                        predicted_value = retrieved_value
                                else:
                                    logging.info(f"  Prop type '{prop_type}' for {player_name} (Pos: {primary_position_type}) doesn't match. Using default prediction 0.")
                                    prediction_status = "Prop Type Mismatch"
                            
                            elif retrieved_value is not None:
                                predicted_value = retrieved_value
                        else:
                            logging.info(f"  Prop type '{prop_type}' not found in mapping for {player_name}. Using default prediction 0.")
                            prediction_status = "Prop Not In Map"
                else:
                    prediction_status = "Player Not Found"

            # Append the data once at the end of the loop with the determined predicted_value
            predicted_props_data.append({
                "date": game_date,
                "player": player_name,
                "team": original_team_field,
                "opponent": original_opponent,
                "prop_type": prop_type,
                "line": original_line,
                "predicted_value": predicted_value,
                "prediction_status": prediction_status
            })

        # Convert to DataFrame and overwrite the original file
        if predicted_props_data:
            final_predictions_df = pd.DataFrame(predicted_props_data)
            
            # Reorder columns to have date and status at the end
            cols = ["date", "player", "team", "opponent", "prop_type", "line", "predicted_value", "prediction_status"]
            final_predictions_df = final_predictions_df[cols]

            # Overwrite the original scraper file with the updated data
            final_predictions_df.to_csv(scraped_csv_path, index=False)
            logging.info(f"✅ Updated predictions have been saved back to {scraped_csv_path}")

            # Also print to console for confirmation
            logging.info("--- Final Player Prop Predictions ---")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                logging.info(f"\n{final_predictions_df.to_string()}")
        else:
            logging.info("No specific player prop predictions could be generated.")

    # Clean up the scraped CSV file (removed as per user request)
    # if os.path.exists(scraped_csv_path):
    #     os.remove(scraped_csv_path)
    #     logging.info(f"Cleaned up temporary file: {scraped_csv_path}")

logging.info("\nScript finished.")