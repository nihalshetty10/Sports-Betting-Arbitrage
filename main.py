import subprocess
import sys
import os

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n--- Running {script_name} ---")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors from {script_name}:\n{result.stderr}")

if __name__ == "__main__":
    run_script("prizepicks_scraper.py")
    run_script("mlb_player_prop_model.py")
    run_script("arbitrage_calc.py")
    run_script("auto_bet_parlays.py")

