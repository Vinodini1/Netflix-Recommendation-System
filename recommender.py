import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/model.pkl")

# Load movie titles
movies = pd.read_csv(
    "data/movie_titles.csv",
    header=None,
    names=["MovieID", "Year", "Title"],
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"

)

def recommend_movies(user_id, n=10):
    recommendations = []

    for movie_id in movies["MovieID"]:
        try:
            predicted_rating = model.predict(user_id, movie_id).est
            recommendations.append((movie_id, predicted_rating))
        except:
            pass

    # Sort by predicted rating
    recommendations.sort(key=lambda x: x[1], reverse=True)

    top_movies = []

    for movie_id, rating in recommendations[:n]:
        title = movies.loc[movies["MovieID"] == movie_id, "Title"].values[0]
        top_movies.append((title, round(rating, 2)))

    return top_movies