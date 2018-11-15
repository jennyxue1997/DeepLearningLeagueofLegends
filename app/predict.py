import features
import json
import tensorflow as tf
from tensorflow.contrib import predictor

def get_winning_team(request, model, ordered_team_games):
    payload = json.loads(request.data)
    blue_team = payload["blue"]["blueTeam"]
    red_team = payload["red"]["redTeam"]
    print(payload)
    test_input_fn = features.get_test_input(blue_team, red_team, ordered_team_games)
    predictions = list(model.predict(input_fn=test_input_fn))
    print(float(predictions[0]["probabilities"][0]))
    return {"blue": float(predictions[0]["probabilities"][0]),
            "red": float(predictions[0]["probabilities"][1])}
