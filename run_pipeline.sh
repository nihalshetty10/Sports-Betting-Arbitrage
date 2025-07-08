#!/bin/bash
echo "Pipeline run at $(date)" >> /Users/nihal/nba_player_prop_model/pipeline.log
cd /Users/nihal/nba_player_prop_model
/Users/nihal/.venv/bin/python main.py >> /Users/nihal/nba_player_prop_model/pipeline.log 2>&1 