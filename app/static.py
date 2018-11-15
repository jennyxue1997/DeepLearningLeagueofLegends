import pandas as pd
import numpy as np 
import features
import json

def convert_csv_to_json():
    df = pd.read_csv("formMatches.csv")
    team_games = features.getTeamGames(df)
    ordered_team_games = features.getOrderedTeamGames(team_games)
    print(ordered_team_games[0])
    # team_games_json = {}
    # for team in ordered_team_games:
    #     # team_games_json[team] = []
    #     if team not in team_games_json:
    #         team_games_json[team] = []
    #     for game in ordered_team_games[team]:
    #         team_games_json[team].append(game.to_dict())
    
    # with open("orderedTeaGames.json", "w") as outfile:
    #     json.dump(team_games_json, outfile)
    # print("Done!")

convert_csv_to_json()