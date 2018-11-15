from flask import Flask, request, jsonify
from flask_cors import CORS
import predict
import features
import tensorflow as tf
from tensorflow.contrib import predictor
import json

app = Flask(__name__)
CORS(app)

@app.route('/', methods=["POST"])
def predict_winning_team():
    return jsonify(predict.get_winning_team(request, model, orderedTeamGames))


if __name__ == "__main__":
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("model.ckpt-1000.meta")
        saver.restore(sess, "model.ckpt-1000")

    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from="model.ckpt-1000")
    model = tf.estimator.DNNClassifier(
        model_dir='model/',
        hidden_units=[15,15,15,15,15,15],
        feature_columns=features.feature_columns,
        n_classes=2,
        label_vocabulary=['B', 'R'],
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.01
        ),
        warm_start_from=ws
    )

    with open("OrderedTeamGames.json", "r") as infile:
        orderedTeamGames = json.load(infile)
    app.run()
