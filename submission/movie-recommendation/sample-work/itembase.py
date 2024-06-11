from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
train_data = pd.read_csv('ydata-ymovies-user-movie-ratings-train-v1_0.txt', sep='\t', header=None)
movies_data = pd.read_csv('ydata-ymovies-mapping-to-movielens-v1_0.txt', sep='\t', header=None, encoding='Latin-1')

# Calculate item-item similarity matrix
def calculate_similarity():
    # Pivot train_data to create user-item matrix
    user_item_matrix = train_data.pivot(index=0, columns=1, values=3).fillna(0)
    
    # Calculate cosine similarity between items
    similarity_matrix = cosine_similarity(user_item_matrix.T)
    
    return similarity_matrix

similarity_matrix = calculate_similarity()

# Get similar items for a given movie_id
def get_similar_items(movie_id, n=10):
    movie_index = movies_data[movies_data[0] == movie_id].index[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_items = similarity_scores[1:n+1]
    similar_item_ids = [movies_data[0][i[0]] for i in similar_items]
    similar_item_titles = [movies_data[1][movies_data[0] == movie_id].values[0] for movie_id in similar_item_ids]
    return similar_item_ids, similar_item_titles

# Render home page
@app.route('/')
def home():
    return render_template('recommend.html')

# API endpoint for getting similar movies
@app.route('/get_similar_movies', methods=['GET'])
def get_similar_movies():
    movie_id = int(request.args.get('movie_id'))
    n = int(request.args.get('n', 10))
    similar_item_ids, similar_item_titles = get_similar_items(movie_id, n)
    result = {
        'movie_id': movie_id,
        'similar_movies': [{'movie_id': movie_id, 'title': title} for movie_id, title in zip(similar_item_ids, similar_item_titles)]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
