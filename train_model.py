import pandas as pd
from surprise import Dataset, Reader, SVD
import joblib

# -----------------------------
# Dataset Paths
# -----------------------------
MOVIE_TITLES_PATH = "data/movie_titles.csv"
RATINGS_PATH = "data/combined_data_1.txt"

# -----------------------------
# Load Movie Titles
# -----------------------------
print("Loading movie titles...")

movies = pd.read_csv(
    MOVIE_TITLES_PATH,
    header=None,
    names=["MovieID", "Year", "Title"],
    engine="python",
    encoding="latin1",
    on_bad_lines="skip"
)

print("Movie titles loaded!")
print(movies.head())

# -----------------------------
# Load Ratings
# -----------------------------
print("\nLoading ratings...")

ratings = []

movie_id = None

with open(RATINGS_PATH, "r") as file:

    for line in file:

        line = line.strip()

        if ":" in line:
            movie_id = int(line.replace(":", ""))

        else:
            customer, rating, date = line.split(",")

            ratings.append(
                [
                    movie_id,
                    int(customer),
                    int(rating),
                    date
                ]
            )

ratings = pd.DataFrame(
    ratings,
    columns=[
        "MovieID",
        "CustomerID",
        "Rating",
        "Date"
    ]
)

print("\nRatings Loaded Successfully!")
print(ratings.head())

# -----------------------------
# Merge Datasets
# -----------------------------
df = ratings.merge(
    movies,
    on="MovieID"
)

print("\nMerged Dataset")
print(df.head())

# -----------------------------
# Build Surprise Dataset
# -----------------------------
reader = Reader(rating_scale=(1,5))

data = Dataset.load_from_df(
    df[["CustomerID","MovieID","Rating"]],
    reader
)

trainset = data.build_full_trainset()

print("\nTraining SVD Model...")

model = SVD()

model.fit(trainset)

print("Model Training Completed!")

joblib.dump(model,"models/model.pkl")

print("\nModel saved successfully!")