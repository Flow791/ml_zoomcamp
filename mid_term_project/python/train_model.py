import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

df_leagues = pd.read_csv('mid_term_project/data/csv/Leagues.csv', index_col=0)

df_leagues.drop(['date', 'matchweek', 'home_team_id', 'away_team_id', 'home_team', 'away_team', 'home_manager', 'away_manager', 'home_xg', 'away_xg', 'home_formation', 'away_formation'], axis=1, inplace=True)

df_leagues['home_team_form'] = df_leagues['home_team_form'].fillna('')
df_leagues['home_team_home_form'] = df_leagues['home_team_home_form'].fillna('')
df_leagues['home_team_average_goals_form'] = df_leagues['home_team_average_goals_form'].fillna(0)
df_leagues['home_team_average_xg_form'] = df_leagues['home_team_average_xg_form'].fillna(0)
df_leagues['home_team_average_goals_form_against'] = df_leagues['home_team_average_goals_form_against'].fillna(0)
df_leagues['home_team_average_xg_form_against'] = df_leagues['home_team_average_xg_form_against'].fillna(0)
df_leagues['away_team_form'] = df_leagues['away_team_form'].fillna('')
df_leagues['away_team_away_form'] = df_leagues['away_team_away_form'].fillna('')
df_leagues['away_team_average_goals_form'] = df_leagues['home_team_average_goals_form'].fillna(0)
df_leagues['away_team_average_xg_form'] = df_leagues['home_team_average_xg_form'].fillna(0)
df_leagues['away_team_average_goals_form_against'] = df_leagues['home_team_average_goals_form_against'].fillna(0)
df_leagues['away_team_average_xg_form_against'] = df_leagues['home_team_average_xg_form_against'].fillna(0)


def calculate_points(results):
    points_map = {'W': 3, 'D': 1, 'L': 0}
    return sum(points_map[char] for char in results)

df_leagues['home_team_form'] = df_leagues['home_team_form'].apply(calculate_points)
df_leagues['away_team_form'] = df_leagues['away_team_form'].apply(calculate_points)

df_leagues['home_team_home_form'] = df_leagues['home_team_home_form'].apply(calculate_points)
df_leagues['away_team_away_form'] = df_leagues['away_team_away_form'].apply(calculate_points)


def get_train_test_split(df: pd.DataFrame, label_name:str = 'home_goals') -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    y_train = df_train[f'{label_name}']
    y_test = df_test[f'{label_name}']

    del df_train[f'{label_name}']
    del df_test[f'{label_name}']
    
    return df_train, df_test, y_train, y_test


def get_X_train_test(df_train:pd.DataFrame, df_test:pd.DataFrame):
    dict_vectorizer = DictVectorizer(sparse=False)
    train_serie_dict = df_train.to_dict(orient='records')
    test_serie_dict = df_test.to_dict(orient='records')

    X_train = dict_vectorizer.fit_transform(train_serie_dict)
    X_test = dict_vectorizer.transform(test_serie_dict)
    
    return X_train, X_test, dict_vectorizer


df_leagues_scores = df_leagues.copy()

df_leagues_scores = df_leagues.copy()
df_leagues_scores['result'] = [0 if x == 0 else 1 if x > 0 else 2 for x in df_leagues_scores['home_goals'] - df_leagues_scores['away_goals']]

df_train_score, df_test_score, y_train_score, y_test_score = get_train_test_split(df_leagues_scores, 'result')

del df_train_score['home_goals']
del df_train_score['away_goals']
del df_test_score['home_goals']
del df_test_score['away_goals']

X_train_score, X_test_score, score_dict_vectorizer = get_X_train_test(df_train_score, df_test_score)

class_counts = Counter(y_train_score)

doubt_weight = (class_counts[0] + class_counts[1] + class_counts[2]) / class_counts[0]
home_weight = (class_counts[0] + class_counts[1] + class_counts[2]) / class_counts[1]
away_weight = (class_counts[0] + class_counts[1] + class_counts[2]) / class_counts[2]

class_weights = {0: doubt_weight, 1: home_weight, 2: away_weight}

result_model = RandomForestClassifier(max_depth=26, 
                                      n_estimators=400,
                                      min_samples_split= 2,
                                      class_weight= class_weights,
                                      random_state=42)
result_model.fit(X_train_score, y_train_score)

output_file = f'mid_term_project/model/model_result.bin'
f_out = open(output_file, 'wb')
pickle.dump((score_dict_vectorizer, result_model), f_out)
f_out.close()

print('Models saved !')