import requests

url = 'http://localhost:9696/predict'
game = {
    "league":"Premier League ",
    "home_team_form":0,
    "home_team_league_pos":1.0,
    "home_team_points_diff":0,
    "home_team_home_form":0,
    "home_team_home_league_pos":1.0,
    "home_team_home_points_diff":0,
    "home_team_average_goals_form":0.0,
    "home_team_average_xg_form":0.0,
    "home_team_average_goals_form_against":0.0,
    "home_team_average_xg_form_against":0.0,
    "away_team_form":0,
    "away_team_league_pos":1.0,
    "away_team_points_diff":0,
    "away_team_away_form":0,
    "away_team_away_league_pos":1.0,
    "away_team_away_points_diff":0,
    "away_team_average_goals_form":0.0,
    "away_team_average_xg_form":0.0,
    "away_team_average_goals_form_against":0.0,
    "away_team_average_xg_form_against":0.0
    }

response = requests.post(url, json=game).json()
print(response)