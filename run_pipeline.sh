#!/bin/bash
echo "Pipeline run at $(date)" >> /Users/nihal/nba_player_prop_model/pipeline.log
cd /Users/nihal/nba_player_prop_model
# Run scrape_stats.py with today's date in MM-DD-YYYY format (for 9:30 AM job)
today=$(date +"%m-%d-%Y")
/Users/nihal/.venv/bin/python prizepicks_scraper.py >> /Users/nihal/nba_player_prop_model/pipeline.log 2>&1
/Users/nihal/.venv/bin/python mlb_player_prop_model.py >> /Users/nihal/nba_player_prop_model/pipeline.log 2>&1
/Users/nihal/.venv/bin/python arbitrage_calc.py >> /Users/nihal/nba_player_prop_model/pipeline.log 2>&1
/Users/nihal/.venv/bin/python auto_bet_parlays.py >> /Users/nihal/nba_player_prop_model/pipeline.log 2>&1

# For 1:15 AM job: run scrape_stats.py and evaluate_props.py for yesterday's date
yesterday=$(date -v-1d +"%m-%d-%Y")
/Users/nihal/.venv/bin/python scrape_stats.py $yesterday >> /Users/nihal/nba_player_prop_model/pipeline.log 2>&1
/Users/nihal/.venv/bin/python evaluate_props.py $yesterday >> /Users/nihal/nba_player_prop_model/pipeline.log 2>&1 
