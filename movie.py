# movie.py

import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

df = None
vectors = None
similarity = None


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def process_genres(genres):
    try:
        genres_list = json.loads(genres)
        return ', '.join(i['name'] for i in genres_list if isinstance(i, dict) and 'name' in i)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error processing genres: {genres}, error: {e}")
        return np.nan

def stem(text):
    ps = PorterStemmer()
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def main(file_path):
    df = load_data(file_path)
    df['overview'] = df['overview'].apply(lambda x: str(x).replace(',', ''))
    df['overview'] = df['overview'].apply(lambda x: str(x).replace('""', ''))
    df['genre'] = df['genres'].apply(process_genres)
    df = df.drop(columns=['genres'], axis=1)
    df['genre'] = df['genre'].apply(lambda x: str(x).replace(',', ''))
    df['combined'] = df['overview'] + df['genre']
    df['key'] = df['keywords'].apply(process_genres)
    df['key'] = df['key'].apply(lambda x: str(x).replace(',', ''))
    df['combined'] = df['combined'] + df['key']
    df = df.iloc[:, [4, 5, 20]]
    df['combined'] = df['combined'].apply(lambda x: str(x).replace('"', ''))
    df['combined'] = df['combined'].apply(lambda x: x.lower())
    df['combined'] = df['combined'].apply(stem)
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    vectors = vectorizer.fit_transform(df['combined']).toarray()
    similarity = cosine_similarity(vectors)
    return df, vectors, similarity

def recommend_movie(movie, df, vectors, similarity):
    movie_index = df[df['original_title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    list1=[]
    for i in movies_list:
        list1.append(df.iloc[i[0]].original_title)
        
    return list1
# Wrapper function definition
file_path = os.path.join(os.path.dirname(__file__), 'data.csv')
df, vectors, similarity = main(file_path)
def recommend_movie_wrapper(movie):   
    return recommend_movie(movie, df, vectors, similarity)
