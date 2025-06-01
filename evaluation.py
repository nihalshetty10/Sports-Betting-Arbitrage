import pandas as pd
from sklearn.metrics import mean_absolute_error

def evaluate_predictions(df):
    # df should have: predicted_mean, actual_stat
    mae = mean_absolute_error(df['actual_stat'], df['predicted_mean'])
    accuracy = (abs(df['predicted_mean'] - df['actual_stat']) < 1).mean()
    print(f"MAE: {mae:.2f}, Accuracy: {accuracy:.2%}")

def evaluate_bets(df):
    # df should have: bet_result (1=win, 0=loss), bet_return
    avg_return = df['bet_return'].mean()
    hit_rate = df['bet_result'].mean()
    print(f"Avg Return: {avg_return:.2%}, Hit Rate: {hit_rate:.2%}")
