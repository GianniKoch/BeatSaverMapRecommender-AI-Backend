import flask
from flask import Flask, request, jsonify
from waitress import serve

from Recommender import Recommender

app = Flask(__name__)
recommender = Recommender(dataset_path='datasets/beatsaversongs.csv')


@app.get("/recommendation")
def get_recommendation():
    song_id = request.args.get('song_id', type=str)
    difficulty = request.args.get('difficulty', type=int)
    characteristic = request.args.get('characteristic', type=int, default=0)
    n_recommendations = request.args.get('n_recommendations', type=int, default=20)
    n_best_tags = request.args.get('n_best_tags', type=int, default=3)

    try:
        recommendations = recommender.recommend(song_id=song_id, difficulty=difficulty, characteristic=characteristic,
                                                n_recommendations=n_recommendations, n_best_tags=n_best_tags)

        response = jsonify(recommendations)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        return {"error": str(e), "parameters": {
            "song_id": song_id,
            "difficulty": difficulty,
            "characteristic": characteristic,
            "n_recommendations": n_recommendations,
            "n_best_tags": n_best_tags
        }}, 400


if __name__ == '__main__':
    host = "0.0.0.0"
    port = 8081
    print(f"Starting server on {host}:{port}")

    serve(app, host=host, port=port)
