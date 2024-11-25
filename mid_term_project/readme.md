# Football Match Outcome Prediction

This project is focused on predicting the outcome of football matches, specifically whether the home team will win or draw or the away team will win. The predictions are made using a Random Forest Classifier, which was trained on historical match data. 

## Project Overview

The goal of this project is to build a model that can predict match outcomes for football leagues based on various factors such as the teams involved, their recent performances, and other relevant data. Instead of predicting exact match scores, the model predicts the general outcome (Home, Away or Draw). 

### Key Features:

- **Data Processing**: The data includes historical match results from various football leagues, with features such as the home and away teams, matchweek, previous results, and more.
- **Modeling**: We used a Random Forest Classifier to classify match outcomes (Home Win, Away Win or Draw).
- **Evaluation**: The model was evaluated using classification metrics like accuracy, precision, recall, and F1 score to assess its ability to predict match outcomes.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/Flow791/ml_zoomcamp
    ```

2. Run Docker:
    ```
    cd project_dir/mid_term_project
    docker build -f env/Dockerfile -t mid_term_project .
    docker run -it --rm -p 9696:9696 mid_term_project   
    ```

3. Run training & test python scripts:
    ```
    source .venv/bin/activate
    python mid_term_project/python/train_model.py
    python mid_term_project/python/test_prediction.py
    ```

## How It Works

1. **Data Preparation**: Historical match data is preprocessed to include features like the teams' previous performances, home and away stats, and more. The target variable is the match result: Home Win (1), Away Win (2) or Draw (0).
   
2. **Training the Model**: A Random Forest Classifier is used to train the model on the historical data, using various hyperparameters.

3. **Prediction**: The trained model predicts whether the home team win or draw, or away team win based on the input data.

## Usage

1. Run Docker:
    ```
    cd project_dir/mid_term_project
    docker build -f env/Dockerfile -t mid_term_project .
    docker run -it --rm -p 9696:9696 mid_term_project   
    ```

2. To use the model for making predictions, you can pass a new match's data through the API or directly use the model in a Python script.

Example usage in Python:
```python
import pickle

# Load the trained model and vectorizer
with open('../mid_term_project/model/model_result.bin', 'rb') as model_file:
    model_home = pickle.load(model_file)

# Sample match data
match_data = {
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

# Predict the outcome
predicted_result = model_home.predict([match_data])

# Evaluation

The model was evaluated on a test set, and the results showed reasonable performance for predicting match outcomes, although improvements can still be made in the prediction of certain outcomes (like draws).

# Key Metrics:
Accuracy: 59%
Precision: 55%
Recall: 53%
F1-score: 49%