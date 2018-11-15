import tensorflow as tf
import numpy as np
import json

feature_columns = [
    tf.feature_column.numeric_column(key='blue-kpm'),
    tf.feature_column.numeric_column(key='red-kpm'),
    tf.feature_column.numeric_column(key='blue-okpm'),
    tf.feature_column.numeric_column(key='red-okpm'),
    tf.feature_column.numeric_column(key='blue-ckpm'),
    tf.feature_column.numeric_column(key='red-ckpm'),
    tf.feature_column.numeric_column(key='blue-teamdragkills'),
    tf.feature_column.numeric_column(key='red-teamdragkills'),
    tf.feature_column.numeric_column(key='blue-oppdragkills'),
    tf.feature_column.numeric_column(key='red-oppdragkills'),
    tf.feature_column.numeric_column(key='blue-teamtowerkills'),
    tf.feature_column.numeric_column(key='red-teamtowerkills'),
    tf.feature_column.numeric_column(key='blue-opptowerkills'),
    tf.feature_column.numeric_column(key='red-opptowerkills'),
    tf.feature_column.numeric_column(key='blue-teambaronkills'),
    tf.feature_column.numeric_column(key='red-teambaronkills'),
    tf.feature_column.numeric_column(key='blue-oppbaronkills'),
    tf.feature_column.numeric_column(key='red-oppbaronkills'),
    tf.feature_column.numeric_column(key='blue-dmgtochampsperminute'),
    tf.feature_column.numeric_column(key='red-dmgtochampsperminute'),
    tf.feature_column.numeric_column(key='blue-earnedgpm'),
    tf.feature_column.numeric_column(key='red-earnedgpm'),
    tf.feature_column.numeric_column(key='blue-cspm'),
    tf.feature_column.numeric_column(key='red-cspm'), 
    tf.feature_column.numeric_column(key='blue-gdat10'),
    tf.feature_column.numeric_column(key='red-gdat10'),
    tf.feature_column.numeric_column(key='blue-xpdat10'),
    tf.feature_column.numeric_column(key='red-xpdat10'), 
    tf.feature_column.numeric_column(key='blue-csdat10'),
    tf.feature_column.numeric_column(key='red-csdat10'),
]

def getTeamGames(formMatches):
    """
    input: csv
    output: map team -> map of dates -> games
    """
    teamGames = {}
    for index, row in formMatches.iterrows():
        if row['date'] != None:
            if row['team'] not in teamGames:
                teamGames[row['team']] = {row['date']: row}
            else:
                teamGames[row['team']][row['date']] = row
    return teamGames

def getOrderedTeamGames(teamGames):
    """
    input: team games
    output: map team -> ordered list of games (by time)
    """
    orderedTeamGames = {}
    for team in teamGames:
        orderedTeamGames[team] = []
        currentTeamGames = teamGames[team]
        keylist = list(currentTeamGames)
        keylist.sort()
        for key in keylist:
            orderedTeamGames[team].append(currentTeamGames[key])
    return orderedTeamGames

def getAvgForm(teamName, numLastForm, headings, orderedTeamGames):
    """
    input: teamName, # games to consider, feature vectors
    output: map of headings (feature columns) -> avg value
    """
    avg_form = {}
    for h in headings:
        avg_form[h] = 0.0
    for i in range(max(0, len(orderedTeamGames[teamName]) - numLastForm), len(orderedTeamGames[teamName])):
        formGame = orderedTeamGames[teamName][i]
        for h in headings:
            try:
                value = float(formGame[h])
                if np.isnan(value):
                    avg_form[h] += 0.0
                else:
                    avg_form[h] += value
            except:
                avg_form[h] += 0.0
    for h in headings:
        avg_form[h] /= numLastForm
    return avg_form

def get_test_input(blueTeam, redTeam, orderedTeamGames, numLastForm=5):
    headings = ['kpm', 'okpm', 'ckpm', 'teamdragkills', 'oppdragkills', 'teamtowerkills', 'opptowerkills', 'teambaronkills', 'oppbaronkills', 'dmgtochampsperminute', 'earnedgpm', 'cspm', 'gdat10', 'xpdat10', 'csdat10']
    avg_form_blue = getAvgForm(blueTeam, numLastForm, headings, orderedTeamGames)
    avg_form_red = getAvgForm(redTeam, numLastForm, headings, orderedTeamGames)
    test_features = {}
    for heading in headings:
        key = 'blue-' + heading
        if key not in test_features:
            test_features[key] = []
        test_features[key].append(avg_form_blue[heading])
        key = 'red-' + heading
        if key not in test_features:
            test_features[key] = []
        test_features[key].append(avg_form_red[heading])  

    np_test_features = {}
    for key in test_features:
        np_test_features[key] = np.array(test_features[key])
    test_labels = ["B"]
    test_labels = np.array(test_labels)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=np_test_features,
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )
    return test_input_fn