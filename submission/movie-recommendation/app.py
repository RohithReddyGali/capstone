from sklearn.neighbors import NearestNeighbors
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)


# Load data
with open('ydata-ymovies-movie-content-descr-v1_0.txt', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.readlines()

# Process the lines as needed, e.g., splitting by tabs or commas
data = []
for line in content:
    # Split line by tabs, commas, or other delimiters as appropriate
    values = line.strip().split('\t')  # Example: assuming tab-separated values
    data.append(values)

# Convert the list of lists into a DataFrame
df = pd.DataFrame(data)
column_mapping = {0: 'Yahoo! Movie ID', 1: 'Title', 2: 'Synopsis', 4: 'Running Time', 5: 'MPAA Rating',
                  6: 'Reasons for MPAA Rating',
                  7: 'Release Date', 8: 'Distributor', 9: 'Poster URL', 10: 'Genres', 11: 'Directors', 12: 'Director IDs',
                  13: 'Crew Members',
                  14: 'Crew IDs', 15: 'Types of Crew', 16: 'Actors', 17: 'Actor IDs', 18: 'Average Critic Rating',
                  19: 'Number of Critic Ratings',
                  20: 'Number of Awards Won', 21: 'Number of Awards Nominated', 22: 'Awards Won', 23: 'Awards Nominated',
                  24: 'Rating from The Movie Mom',
                  25: 'Review from The Movie Mom', 26: 'Review Summaries', 27: 'Anonymized Review Owners',
                  28: 'Captions from Trailers/Clips',
                  29: "Greg's Preview URL", 30: "DVD Review URL", 31: 'Global Non-Personalized Popularity (GNPP)',
                  32: 'Average User Rating',
                  33: 'Number of Users Who Rated'}

df.rename(columns=column_mapping, inplace=True)
df = df.replace('\\N', '')
df = df.drop(['Synopsis', 'Running Time', 'MPAA Rating',
              'Reasons for MPAA Rating',
              'Release Date', 'Distributor', 'Poster URL',  'Genres',  'Directors',  'Director IDs',
              'Crew Members',
              'Crew IDs',  'Types of Crew',  'Actors',  'Actor IDs',  'Average Critic Rating',
              'Number of Critic Ratings',
              'Number of Awards Won',  'Number of Awards Nominated',  'Awards Won',  'Awards Nominated',
              'Rating from The Movie Mom',
              'Review from The Movie Mom',  'Review Summaries',  'Anonymized Review Owners',
              'Captions from Trailers/Clips',
              "Greg's Preview URL",  "DVD Review URL",  'Global Non-Personalized Popularity (GNPP)',
              'Average User Rating',
              'Number of Users Who Rated'], axis=1)

# Load data
ratings_train = pd.read_csv('ydata-ymovies-user-movie-ratings-train-v1_0.txt',
                            delimiter='\t', names=['user_id', 'movie_id', 'ratingfor12', 'rating'])
ratings_train = ratings_train.drop_duplicates(
    subset=['user_id', 'movie_id'])  # Remove duplicate entries
ratings_train = ratings_train.drop(['ratingfor12'], axis=1)
ratings_test = pd.read_csv('ydata-ymovies-user-movie-ratings-test-v1_0.txt',
                           delimiter='\t', names=['user_id', 'movie_id', 'ratingfor12', 'rating'])
ratings_test = ratings_test.drop(['ratingfor12'], axis=1)

# LOad Data from Item-item based
train_data = pd.read_csv(
    'ydata-ymovies-user-movie-ratings-train-v1_0.txt', sep='\t', header=None)
#movies_data = pd.read_csv('/kaggle/input/yahoomovies/ydata-ymovies-mapping-to-movielens-v1_0.txt', sep='\t', header=None)
with open('ydata-ymovies-mapping-to-movielens-v1_0.txt', 'r', encoding='utf-8', errors='ignore') as f:
    contents = f.readlines()
# Process the lines as needed, e.g., splitting by tabs or commas
data1 = []
for line in contents:
    # Split line by tabs, commas, or other delimiters as appropriate
    values = line.strip().split('\t')  # Example: assuming tab-separated values
    data1.append(values)

# Convert the list of lists into a DataFrame
movies_data = pd.DataFrame(data1)

column_mapping = {0: 'user_id', 1: 'movie_id', 2: 'Rate', 3: 'Rating'}
train_data.rename(columns=column_mapping, inplace=True)
# train_data.head()
column_mapping = {0: 'movie_id', 1: 'Title', 2: 'movielens_movie_id'}
movies_data.rename(columns=column_mapping, inplace=True)
# movies_data.head()

user_ratings_df = train_data
movie_info_df = movies_data


# -----------------------------------------


def get_movie_recommendations(movie_id, user_ratings_df, movie_info_df, k=5):
    """
    Get movie recommendations based on item-item collaborative filtering using k-nearest neighbors.

    Args:
        movie_id (str): Movie ID for which to get recommendations.
        user_ratings_df (pd.DataFrame): DataFrame containing user ratings data.
        movie_info_df (pd.DataFrame): DataFrame containing movie information data.
        k (int, optional): Number of neighbors to consider. Defaults to 5.

    Returns:
        list: List of movie IDs recommended for the given movie ID.
    """
    # Create pivot table for item-item based recommendation
    pivot_table = user_ratings_df.pivot_table(
        index='movie_id', columns='user_id', values='Rate', fill_value=0)

    # Instantiate and fit the k-nearest neighbors model
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(pivot_table)
    if len(movie_info_df[movie_info_df['movie_id'] == str(movie_id)]) > 0:
        # Find the index of the selected movie in the pivot table
        selected_movie_index = movie_info_df[movie_info_df['movie_id'] == str(
            movie_id)].index[0]

        # Find the k-nearest neighbors to the selected movie
        distances, indices = knn.kneighbors(
            pivot_table.iloc[selected_movie_index].values.reshape(1, -1), n_neighbors=k+1)

        # Extract the movie IDs of the nearest neighbors (excluding the selected movie itself)
        nearest_movie_ids = pivot_table.index[indices.flatten()[1:]].tolist()

        return nearest_movie_ids


# ------------------------------------------------------------
# Perform user-based collaborative filtering
# Step 1: Compute user-item matrix
user_item_matrix = ratings_train.pivot(
    index='user_id', columns='movie_id', values='rating').fillna(0)

# Step 2: Compute user similarity matrix
user_similarity = cosine_similarity(user_item_matrix)

# Step 3: Make recommendations for a user


def get_top_n_recommendations(user_id, n):
    if user_id not in user_item_matrix.index:
        return []  # Return empty list if user_id not found in ratings_train dataset

    user_index = user_item_matrix.index.get_loc(user_id)
    similar_users_indices = user_similarity[user_index].argsort()[::-1][1:]
    recommended_movies = []
    for user_idx in similar_users_indices:
        similar_user_id = user_item_matrix.index[user_idx]
        similar_user_movies = ratings_train[ratings_train['user_id']
                                            == similar_user_id]['movie_id']
        for movie_id in similar_user_movies:
            if movie_id not in ratings_train[ratings_train['user_id'] == user_id]['movie_id'].values:
                recommended_movies.append(movie_id)
            if len(recommended_movies) == n:
                break
        if len(recommended_movies) == n:
            break
    return recommended_movies

# -------------------------------------------------


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        type = str(request.form['recommendation_type'])
        if type == 'item_based':
            movie_id = int(request.form['movie_id'])
            s_id = movie_id
            recommended_movies = get_movie_recommendations(
                s_id, user_ratings_df, movie_info_df, k=5)
            #recommended_movies = get_top_n_recommendations(user_id, n)
            pdf = pd.DataFrame(columns=['Yahoo! Movie ID', 'Title'])

            # when no recommended moview just render the empty template
            if recommended_movies is None:
                return render_template('index.html')

            # Loop through the recommended movies list
            for i in recommended_movies:
                # Filter the original DataFrame based on movie ID
                filtered_movies = df[df['Yahoo! Movie ID'] == str(i)]

                # Append the filtered movies to the pdf DataFrame, selecting only the 'Yahoo! Movie ID' and 'Title' columns
                pdf = pdf.append(
                    filtered_movies[['Yahoo! Movie ID', 'Title']], ignore_index=True)
            print(pdf)
            # Reset the index of the pdf DataFrame
            pdf.reset_index(drop=True, inplace=True)
        else:
            user_id = int(request.form['user_id'])
            s_id = user_id
            recommended_movies = get_top_n_recommendations(s_id, 5)
            print(recommended_movies)
            #recommended_movies = get_top_n_recommendations(user_id, n)
            pdf = pd.DataFrame(columns=['Yahoo! Movie ID', 'Title'])

            # when no recommended moview just render the empty template
            if recommended_movies is None or len(recommended_movies) <= 0:
                return render_template('index.html')

            # Loop through the recommended movies list
            for i in recommended_movies:
                # Filter the original DataFrame based on movie ID
                filtered_movies = df[df['Yahoo! Movie ID'] == str(i)]

                # Append the filtered movies to the pdf DataFrame, selecting only the 'Yahoo! Movie ID' and 'Title' columns
                pdf = pdf.append(
                    filtered_movies[['Yahoo! Movie ID', 'Title']], ignore_index=True)
            print(pdf)
            # Reset the index of the pdf DataFrame
            pdf.reset_index(drop=True, inplace=True)

        return render_template('index.html', user_id=s_id, recommended_movies=pdf)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
