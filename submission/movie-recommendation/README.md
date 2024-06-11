# Movie Recommendation System
This is a movie recommendation system implemented using Flask, scikit-learn, and pandas libraries in Python.

## Description
The movie recommendation system provides two types of recommendations: item-based and user-based.

### Item-based recommendation: 
Given a movie ID, it recommends similar movies based on item-item collaborative filtering. It uses the k-nearest neighbors algorithm with cosine similarity as the distance metric.

### User-based recommendation: 
Given a user ID, it recommends movies that are popular among users who have similar movie preferences. It computes the similarity between users based on their ratings using cosine similarity and suggests movies that similar users have rated positively.

### The system utilizes the following datasets:

* #### ydata-ymovies-movie-content-descr-v1_0.txt: 
Contains movie information such as title, genres, directors, and actors.

* ####  ydata-ymovies-user-movie-ratings-train-v1_0.txt: 
Contains user ratings for movies in the training set.

* ####  ydata-ymovies-user-movie-ratings-test-v1_0.txt:
Contains user ratings for movies in the test set.

* ####  ydata-ymovies-mapping-to-movielens-v1_0.txt: 
Maps movie IDs in the system to MovieLens movie IDs.

## Installation

To run the movie recommendation system, you need to follow these steps:

* Install Python (version 3.7 or higher) and pip (if not already installed).
* Install the required dependencies by running the following command in the project directory:

``` 
pip install -r requirements.txt
```
* Start the Flask application by running the following command:

```
python app.py
```
* If any of the required packages are not installed, you can use the following command to install them:

```
pip install <package_name>
```
Replace <package_name> with the name of the package that needs to be installed.

* Access the movie recommendation system in your web browser at 
#### http://localhost:5000.

## Usage

* Upon accessing the web application, you will see a form where you can choose the recommendation type (item-based or user-based) and provide the required input (movie ID or user ID).
* Select the recommendation type and enter the movie ID or user ID.
* Click the "Get Recommendations" button to get the movie recommendations.
* The system will display the recommended movies along with their Yahoo! Movie IDs and titles.
* If there are no recommendations for the given input, "No Recommendation" will be displayed.

Note: The movie IDs and user IDs should correspond to the dataset used by the system.

## Sample Inputs
To use the movie recommendation system, follow these steps:

#### User-Based Filtering:

* Select "User-Based" from the dropdown menu.
* In the "User ID" input field, enter a valid user ID (e.g., 1, 53, 67, 45, etc.).
* Click the "Get Recommendations" button.

#### Item-Based Filtering:

* Select "Item-Based" from the dropdown menu.
* In the "Movie ID" input field, enter a valid movie ID (e.g., 1800361191, 1800249828, 1802956884, 1800354250, etc.).
* Click the "Get Recommendations" button.

The system will generate a list of recommended movies based on the selected filtering method and the provided input.

## Acknowledgements
The movie recommendation system is built using scikit-learn, Flask, and pandas libraries. The dataset used in this project is provided by Yahoo! Research.
