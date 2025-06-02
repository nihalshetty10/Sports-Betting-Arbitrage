#pip install pybaseball
import pandas as pd
import pybaseball
from pybaseball import batting_stats, statcast_batter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Get player list for 2023 and 2024
players_23 = batting_stats(2023)
players_24 = batting_stats(2024)
players = pd.concat([players_23, players_24]).drop_duplicates(subset=['IDfg'])


# 3. Prepare data
stats = ['runs', 'total_bases', 'hits', 'home_runs', 'strikeouts', 'rbi']
stat_map = {
    'runs': 'r',
    'total_bases': 'tb',
    'hits': 'h',
    'home_runs': 'hr',
    'strikeouts': 'so',
    'rbi': 'rbi',
}
data = []
for idx, row in players.iterrows():
    player_id = row['IDfg']
    try:
        # Get all games for 2023 and 2024
        logs_23 = statcast_batter('2023-03-01', '2023-11-01', player_id)
        logs_24 = statcast_batter('2024-03-01', '2024-11-01', player_id)
        logs = pd.concat([logs_23, logs_24])
        if logs.empty or len(logs) < 11:
            continue
        logs = logs.sort_values('game_date')
        last_10 = logs.tail(10)
        historical = logs.iloc[:-10]
        features = {}
        for stat in stats:
            stat_col = stat_map[stat]
            last_10_avg = last_10[stat_col].mean() if stat_col in last_10.columns and not last_10.empty else 0
            hist_avg = historical[stat_col].mean() if stat_col in historical.columns and not historical.empty else 0
            features[f"{stat}_weighted"] = 0.65 * last_10_avg + 0.35 * hist_avg
        # Targets: next game's stats
        target_row = logs.iloc[-1]
        for stat in stats:
            stat_col = stat_map[stat]
            features[f"target_{stat}"] = target_row[stat_col] if stat_col in target_row else 0
        data.append(features)
    except Exception as e:
        print(f"Error for player {player_id}: {e}")

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
def predict_player_stats(player_id, trained_models, players_df, stat_map, stats):
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
            stat_col = stat_map[stat]
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
example_player_id = players['IDfg'].iloc[0] # Using the first player from the fetched list as an example
predicted_stats = predict_player_stats(example_player_id, trained_models, players, stat_map, stats)

if predicted_stats:
    print(f"\nPredicted stats for player ID {example_player_id}:")
    for stat, pred_value in predicted_stats.items():
        print(f"  {stat.replace('_weighted', '').capitalize()}: {pred_value}") 