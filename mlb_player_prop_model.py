#pip install pybaseball, unidecode, thefuzz, baseball-id
import pandas as pd
import pybaseball
from pybaseball import batting_stats, statcast_batter, playerid_lookup, statcast_pitcher
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
import time 
import datetime 
import numpy as np 
import subprocess 
import os 
import logging
from unidecode import unidecode 
from thefuzz import process 
from baseball_id import Lookup 
import re
from bs4 import BeautifulSoup
import string
import functools
import urllib.error

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

# Load baseball-id master table at startup
try:
    BASEBALL_ID_MASTER = Lookup.master.copy()
    BASEBALL_ID_MASTER['fullName'] = (
        BASEBALL_ID_MASTER['mlb_name'].fillna('')
        .apply(lambda x: unidecode(str(x)).strip())
    )
except Exception as e:
    BASEBALL_ID_MASTER = None
    logging.error(f"Could not load baseball-id master table: {e}")

def chadwick_player_lookup(player_name):
    """
    Looks up a player's MLB ID by name using the Chadwick Bureau data.
    Returns the MLB ID if a confident match is found, else None.
    """
    if BASEBALL_ID_MASTER is None:
        logging.warning("Chadwick Bureau master table not loaded, cannot perform player lookup.")
        return None

    player_name_norm = unidecode(player_name).lower().strip()

    # Use fuzzy matching to find the best match from the 'fullName' column
    # The 'fullName' was created from 'mlb_name' at startup
    choices = BASEBALL_ID_MASTER['fullName'].dropna().unique()
    best_match = process.extractOne(player_name_norm, choices, score_cutoff=85)

    if best_match:
        matched_name, score = best_match
        logging.info(f"[Chadwick] Fuzzy matched '{player_name}' with '{matched_name}' (Score: {score})")
        
        # Get the row for the matched player
        player_rows = BASEBALL_ID_MASTER[BASEBALL_ID_MASTER['fullName'] == matched_name]
        
        if not player_rows.empty:
            # Take the first match if multiple have the same name
            player_id = player_rows.iloc[0]['mlb_id']
            if pd.notna(player_id):
                logging.info(f"[Chadwick] Found MLB ID: {int(player_id)}")
                return int(player_id)
            else:
                logging.warning(f"[Chadwick] Matched player '{matched_name}' but they have no MLB ID.")
                return None

    logging.warning(f"[Chadwick] Could not find a confident match for '{player_name}'.")
    return None

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

def _get_api_player_ids(years):
    """Gets a set of player IDs from the MLB Stats API for given years."""
    player_ids = set()
    for group in ['hitting', 'pitching']:
        for year in years:
            try:
                url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&group={group}&season={year}"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                if data and 'stats' in data and len(data['stats']) > 0 and 'splits' in data['stats'][0]:
                    for entry in data['stats'][0]['splits']:
                        if entry.get('player') and 'id' in entry['player']:
                            player_ids.add(entry['player']['id'])
            except requests.exceptions.RequestException as e:
                logging.error(f"Could not fetch {group} season stats for {year}: {e}")
    logging.info(f"Found {len(player_ids)} unique player IDs from MLB API for years {years}.")
    return player_ids

def _get_statcast_player_ids(days=30):
    """Gets a set of player IDs from Statcast data for the last X days."""
    logging.info(f"Fetching player IDs from Statcast for the last {days} days... (This may take a moment)")
    today = datetime.datetime.now()
    start_date = (today - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    player_ids = set()
    try:
        hitters = statcast_batter(start_date, end_date)
        if not hitters.empty:
            player_ids.update(hitters['batter'].dropna().astype(int).unique())
        pitchers = statcast_pitcher(start_date, end_date)
        if not pitchers.empty:
            player_ids.update(pitchers['pitcher'].dropna().astype(int).unique())
    except Exception as e:
        logging.error(f"Could not fetch Statcast data: {e}")
    logging.info(f"Found {len(player_ids)} unique player IDs from recent Statcast data.")
    return player_ids

def build_master_player_map(api_years):
    """Builds a comprehensive player map from multiple sources."""
    logging.info("--- Building Master Player Database ---")
    
    # 1. Gather player IDs from all sources
    api_ids = _get_api_player_ids(api_years)
    statcast_ids = _get_statcast_player_ids()
    all_ids = list(api_ids.union(statcast_ids))
    logging.info(f"Found a total of {len(all_ids)} unique player IDs from all sources.")
    
    # 2. Enrich every player with full details
    player_info_map = {}
    logging.info("Enriching all players with details (name, team, position)...")
    for i, player_id in enumerate(all_ids):
        if i > 0 and i % 100 == 0:
            logging.info(f"  ...processed {i} of {len(all_ids)} players.")
        details = get_player_details(player_id)
        if details:
            player_info_map[player_id] = details
        time.sleep(0.02) # Rate limiting
        
    logging.info("--- Master Player Database Build Complete ---")
    return player_info_map

# 1. Build the master player database at the start
training_years = [year_minus_1, year_minus_2, year_plus_0] # Define years for historical API search
player_info_map = build_master_player_map(training_years)

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

def calculate_prediction_features(historical_logs, stats_to_process, stat_map):
    """Calculates weighted features from a DataFrame of historical game logs."""
    features = {}
    for stat_name in stats_to_process:
        api_key = stat_map.get(stat_name)
        if api_key and api_key in historical_logs.columns:
            # Rolling 10-game average from the most recent season in historical logs
            latest_year_in_logs = historical_logs['game_date'].dt.year.max()
            if latest_year_in_logs:
                latest_year_data = historical_logs[historical_logs['game_date'].dt.year == latest_year_in_logs][api_key]
                rolling_10_logs = latest_year_data.rolling(window=10, min_periods=1).mean().iloc[-1] if not latest_year_data.empty else np.nan
            else:
                rolling_10_logs = np.nan

            # Rolling 5-game average
            rolling_5_avg = historical_logs[api_key].rolling(window=5, min_periods=1).mean().iloc[-1] if not historical_logs.empty else np.nan

            # Rolling 3-game average
            rolling_3_avg = historical_logs[api_key].rolling(window=3, min_periods=1).mean().iloc[-1] if not historical_logs.empty else np.nan

            # Overall historical average
            hist_avg = historical_logs[api_key].mean() if not historical_logs.empty else 0

            # Combine recent averages using specified weights, ignoring NaN values
            recent_averages_with_weights = []
            if pd.notna(rolling_10_logs):
                recent_averages_with_weights.append((rolling_10_logs, 0.25))
            if pd.notna(rolling_5_avg):
                recent_averages_with_weights.append((rolling_5_avg, 0.35))
            if pd.notna(rolling_3_avg):
                recent_averages_with_weights.append((rolling_3_avg, 0.4))

            if recent_averages_with_weights:
                averages = [item[0] for item in recent_averages_with_weights]
                weights = [item[1] for item in recent_averages_with_weights]
                normalized_weights = np.array(weights) / np.sum(weights)
                recent_weighted_avg = np.average(averages, weights=normalized_weights)
            else:
                recent_weighted_avg = 0

            # Create the final weighted feature
            features[f"{stat_name}_weighted_feature"] = 0.75 * recent_weighted_avg + 0.25 * hist_avg
        else:
            features[f"{stat_name}_weighted_feature"] = 0
    return features

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
        features = calculate_prediction_features(historical_logs, stats_to_process, api_stat_map_current)

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

def predict_stats_from_br(player_name, prop_type, trained_hitting_models, trained_pitching_models, hitting_stats, pitching_stats, team_abbr=None):
    """Fallback prediction method using Baseball-Reference game logs, now with team roster lookup and fuzzy matching."""
    logging.info(f"[BR Fallback] Attempting to predict stats for '{player_name}' using Baseball-Reference.")

    is_pitcher = prop_type in ["Pitcher Strikeouts", "Earned Runs Allowed"]
    player_type_code = 'p' if is_pitcher else 'b'
    stats_to_process = pitching_stats if is_pitcher else hitting_stats
    trained_models = trained_pitching_models if is_pitcher else trained_hitting_models
    feature_cols = [f'{stat}_weighted_feature' for stat in stats_to_process]

    br_stat_map = {
        'hits': 'H', 'runs': 'R', 'total_bases': 'TB', 'strikeouts': 'SO',
        'earned_runs_allowed': 'ER'
    }

    br_key = None
    norm_name = unidecode(player_name).lower()
    # 1. Try fuzzy search first
    try:
        player_lookup_df = playerid_lookup(player_name.split()[-1], player_name.split()[0], fuzzy=True)
        if not player_lookup_df.empty:
            player_lookup_df = player_lookup_df[player_lookup_df['mlb_played_last'].notna()].sort_values('mlb_played_last', ascending=False)
            if not player_lookup_df.empty:
                br_key = player_lookup_df.iloc[0]['key_bbref']
                logging.info(f"[BR Fallback] Found BR key '{br_key}' for '{player_name}' via fuzzy search.")
    except Exception as e:
        logging.error(f"[BR Fallback] Error looking up player '{player_name}': {e}")

    # 2. If fuzzy search fails, use team roster mapping (with fuzzy fallback)
    if not br_key and team_abbr:
        player_map, team_to_names, all_names = build_br_roster_player_map()
        abbr = team_abbr.strip().upper()
        key = (norm_name, abbr)
        logging.info(f"[BR Fallback] Attempting exact roster lookup for key: {key}")
        br_key = player_map.get(key)
        if br_key:
            logging.info(f"[BR Fallback] Found BR key '{br_key}' for '{player_name}' on team '{abbr}' via roster scrape.")
        else:
            # Fuzzy match on team roster
            team_names = team_to_names.get(abbr, [])
            if team_names:
                names_only = [n for n, _ in team_names]
                best = process.extractOne(norm_name, names_only, score_cutoff=80)
                logging.info(f"[BR Fallback] Fuzzy matching '{norm_name}' on team '{abbr}' roster: candidates={names_only}")
                if best:
                    idx = names_only.index(best[0])
                    br_key = team_names[idx][1]
                    logging.info(f"[BR Fallback] Fuzzy matched '{player_name}' to '{best[0]}' (score {best[1]}) on team '{abbr}'. BR key: {br_key}")
            # Fuzzy match on all rosters
            if not br_key:
                all_names_only = [n for n, _, _ in all_names]
                best = process.extractOne(norm_name, all_names_only, score_cutoff=80)
                logging.info(f"[BR Fallback] Fuzzy matching '{norm_name}' on all rosters: candidates={all_names_only}")
                if best:
                    idx = all_names_only.index(best[0])
                    br_key = all_names[idx][1]
                    abbr = all_names[idx][2]
                    logging.info(f"[BR Fallback] Fuzzy matched '{player_name}' to '{best[0]}' (score {best[1]}) on team '{abbr}'. BR key: {br_key}")
        if not br_key:
            logging.warning(f"[BR Fallback] Could not find '{player_name}' on team '{abbr}' in roster scrape (even with fuzzy match).")

    if not br_key:
        logging.error(f"[BR Fallback] FINAL MISS: '{player_name}' (normalized: '{norm_name}') team_abbr: '{team_abbr}'")
        return "Player Not Found on BR", None

    all_logs = []
    for year in [year_minus_2, year_minus_1, year_plus_0]:
        for attempt in range(3):  # Try up to 3 times
            try:
                url = f"https://www.baseball-reference.com/players/gl.fcgi?id={br_key}&t={player_type_code}&year={year}"
                tables = pd.read_html(url)
                df = next((tbl for tbl in tables if 'Date' in tbl.columns and 'Rk' in tbl.columns), None)
                if df is not None:
                    df.columns = df.columns.get_level_values(-1)
                    df = df[df['Rk'].ne('Rk')].dropna(subset=['Date'])
                    df['year'] = year
                    all_logs.append(df)
                    logging.info(f"[BR Fallback] Scraped {len(df)} game logs for {year}.")
                time.sleep(5)  # Increase delay between requests
                break  # Success, break retry loop
            except Exception as e:
                # Check for HTTP 429
                if hasattr(e, 'code') and e.code == 429:
                    logging.warning(f"[BR Fallback] 429 Too Many Requests for {player_name} {year}. Waiting before retry...")
                    time.sleep(60)  # Wait a minute before retrying
                else:
                    logging.warning(f"[BR Fallback] Could not scrape gamelogs for {year} for '{player_name}': {e}")
                    break

    if not all_logs:
        return "No Logs on BR", None

    logs = pd.concat(all_logs)
    for stat, col in br_stat_map.items():
        if col in logs.columns:
            logs[col] = pd.to_numeric(logs[col], errors='coerce').fillna(0)

    def process_br_date(row):
        date_str = str(row['Date']).split('(')[0].strip()
        year_str = str(row['year'])
        if year_str in date_str:
            return pd.to_datetime(date_str, errors='coerce')
        else:
            return pd.to_datetime(f"{year_str} {date_str}", errors='coerce')

    logs['game_date'] = logs.apply(process_br_date, axis=1)
    logs.dropna(subset=['game_date'], inplace=True)
    logs = logs.sort_values('game_date').reset_index(drop=True)

    if logs.empty:
        return "No Logs on BR after date parsing", None

    features = calculate_prediction_features(logs, stats_to_process, br_stat_map)
    X_predict = pd.DataFrame([features]).reindex(columns=feature_cols, fill_value=0)

    predictions = {}
    for stat in stats_to_process:
        if stat in trained_models:
            model = trained_models[stat]
            prediction = model.predict(X_predict[[f'{stat}_weighted_feature']])[0]
            predictions[stat] = max(0, round(prediction, 2))
        else:
            predictions[stat] = 0
    
    return "Success from BR", predictions

def find_player_id_by_name(player_name, team_abbreviation, player_info_map, team_map):
    """Finds the best player ID match using team and fuzzy name matching (Primary Search)."""
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
        logging.warning(f"{log_message} Searching all players as a fallback for primary search.")
        possible_players = {info['fullName']: pid for pid, info in player_info_map.items() if info.get('fullName')}
        search_scope_message = "across all teams"

    if not possible_players:
        logging.error("Player info map is completely empty. Cannot find any player.")
        return None
        
    best_match = process.extractOne(player_name_norm, possible_players.keys(), score_cutoff=85)

    if best_match:
        matched_name, score = best_match[0], best_match[1]
        player_id = possible_players[matched_name]
        logging.info(f"Primary search matched '{player_name}' with '{matched_name}' (Score: {score}) -> Player ID: {player_id}")
        return player_id
    else:
        logging.warning(f"Primary search could not find a confident match for '{player_name}' {search_scope_message}.")
        return None

# --- Baseball-Reference Team Roster Scraping ---
BR_TEAMS_URL = "https://www.baseball-reference.com/teams/"

def get_br_team_abbrs():
    """Scrape Baseball-Reference teams page to get all current MLB team abbreviations and their roster URLs."""
    res = requests.get(BR_TEAMS_URL)
    soup = BeautifulSoup(res.text, 'html.parser')
    teams = {}
    for row in soup.select('table#teams_active tbody tr'):  # Only active teams
        link = row.find('a')
        if link and '/teams/' in link['href']:
            abbr = link['href'].split('/')[2]
            team_name = link.text.strip()
            teams[abbr] = {
                'name': team_name,
                'roster_url': f"https://www.baseball-reference.com/teams/{abbr}/2024-roster.shtml"
            }
    return teams

@functools.lru_cache(maxsize=1)
def build_br_roster_player_map():
    """Build a mapping from (player name, team abbr) to BR ID and a team->names mapping for fuzzy search."""
    teams = get_br_team_abbrs()
    player_map = {}
    team_to_names = {}
    for abbr, info in teams.items():
        url = info['roster_url']
        team_to_names[abbr] = []
        try:
            res = requests.get(url)
            soup = BeautifulSoup(res.text, 'html.parser')
            for row in soup.select('table#team_roster tbody tr'):
                link = row.find('a')
                if link and '/players/' in link['href']:
                    player_name = link.text.strip()
                    br_id = link['href'].split('/')[-1].replace('.shtml', '')
                    norm_name = unidecode(player_name).lower()
                    player_map[(norm_name, abbr)] = br_id
                    team_to_names[abbr].append((norm_name, br_id))
        except Exception as e:
            logging.warning(f"Could not scrape roster for {abbr}: {e}")
    # Also build a global list for all rosters
    all_names = []
    for abbr, names in team_to_names.items():
        all_names.extend([(n, br_id, abbr) for n, br_id in names])
    return player_map, team_to_names, all_names

# --- END Baseball-Reference Team Roster Scraping ---

def save_br_scrape_results(results, output_path=None):
    """Save a list of dicts (BR fallback logs/results) to a CSV file in the project root."""
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "br_scrape_results.csv")
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logging.info(f"[BR Fallback] Saved BR scrape results to {output_path}")
    else:
        logging.info("[BR Fallback] No BR scrape results to save.")

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

        br_scrape_results = []

        for index, row in scraped_props_df.iterrows():
            player_name = row['player']
            prop_type = row['prop_type']
            original_line = row['line']
            original_team_field = row['team']
            original_opponent = row['opponent']
            
            predicted_value = 0.0 # Default to 0.0, ensures no NaN values
            prediction_status = "Unknown Error" # Default status

            # Attempt to parse team abbreviation
            team_abbreviation = original_team_field.split(' - ')[0].strip()

            # Skip combined player props
            if '+' in player_name:
                prediction_status = "Combined Player Prop"
            else:
                # --- Stage 1: Try MLB API ---
                player_id = find_player_id_by_name(player_name, team_abbreviation, player_info_map, TEAM_ABBREVIATION_MAP)
                prediction_made = False

                if player_id:
                    # Ensure player details are in the map before predicting
                    if player_id not in player_info_map:
                        details = get_player_details(player_id)
                        if details:
                            player_info_map[player_id] = details
                        else:
                            player_id = None # Cannot proceed if details are not fetchable

                if player_id:
                    status, predicted_stats = predict_player_stats(player_id, trained_hitting_models, trained_pitching_models, api_hitting_stat_map, api_pitching_stat_map, hitting_stats, pitching_stats, player_info_map)
                    
                    if status == "Success" and isinstance(predicted_stats, dict):
                        prediction_status = "Success (API)"
                        prediction_made = True
                        # Extract the predicted value for the specific prop
                        internal_stat_name = prop_type_to_stat_name.get(prop_type)
                        if internal_stat_name:
                            predicted_value = predicted_stats.get(internal_stat_name, 0.0)
                        else:
                            prediction_status = "Prop Not In Map"
                    else:
                        logging.warning(f"MLB API prediction failed for {player_name} with status '{status}'. Falling back to BR.")
                
                # --- Stage 2: Fallback to Baseball-Reference ---
                if not prediction_made:
                    br_status, br_predictions = predict_stats_from_br(player_name, prop_type, trained_hitting_models, trained_pitching_models, hitting_stats, pitching_stats, team_abbreviation)

                    if br_status == "Success from BR" and isinstance(br_predictions, dict):
                        prediction_status = br_status
                        # Extract the predicted value for the specific prop
                        internal_stat_name = prop_type_to_stat_name.get(prop_type)
                        if internal_stat_name:
                            predicted_value = br_predictions.get(internal_stat_name, 0.0)
                        else:
                            prediction_status = "Prop Not In Map"
                    else:
                        # If both API and BR fail, mark as not found
                        prediction_status = f"Player Not Found ({br_status})"
                    # Save BR fallback attempt
                    br_scrape_results.append({
                        "player": player_name,
                        "team": team_abbreviation,
                        "prop_type": prop_type,
                        "status": br_status,
                        "predicted_value": predicted_value
                    })

            # Append the data once at the end of the loop
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

            # Add residual column (prediction - actual)
            final_predictions_df['residual'] = final_predictions_df['predicted_value'] - final_predictions_df['line']

            # Calculate rolling std per prop_type and set prediction column
            final_predictions_df['rolling_sd'] = np.nan
            final_predictions_df['prediction'] = np.nan
            for prop in final_predictions_df['prop_type'].unique():
                mask = final_predictions_df['prop_type'] == prop
                # Rolling std of residual, window=10, min_periods=3
                rolling_std = final_predictions_df.loc[mask, 'residual'].rolling(window=10, min_periods=3).std()
                final_predictions_df.loc[mask, 'rolling_sd'] = rolling_std.values
                # Set prediction: normalized if rolling_sd is not NaN, else raw predicted_value
                pred = final_predictions_df.loc[mask, 'predicted_value'] / rolling_std
                # Where rolling_sd is NaN, use predicted_value
                pred = pred.where(~rolling_std.isna(), final_predictions_df.loc[mask, 'predicted_value'])
                final_predictions_df.loc[mask, 'prediction'] = pred.values

            # Save with new columns as well
            final_predictions_df.to_csv(scraped_csv_path, index=False)
            final_predictions_df.to_csv('mlb_predictions_with_analytics.csv', index=False)

            # Also print to console for confirmation
            logging.info("--- Final Player Prop Predictions ---")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                logging.info(f"\n{final_predictions_df.to_string()}")
        else:
            logging.info("No specific player prop predictions could be generated.")

        # After all predictions:
        save_br_scrape_results(br_scrape_results)

    logging.info("\nScript finished.")