# app.py
import logging
from flask import Flask, request, jsonify
from movie import recommend_movie_wrapper
import time

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    return "jai ho"

@app.route('/movie', methods=['POST'])
def movie():
    movie = request.json.get('movie')
    if not movie:
        return jsonify({'error': 'No movie title provided'}), 400
    
    try:
        recommendations = recommend_movie_wrapper(movie)
        if(recommendations is not None):
             return jsonify({'recommendations': recommendations})
        return jsonify({'recommendations': "not get"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(port=5000,debug=True)

