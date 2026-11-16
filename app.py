import streamlit as st
from recommender import recommend_movies

st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)
st.sidebar.title("Netflix Recommender")
st.sidebar.write("Machine Learning Project")
st.sidebar.write("Algorithm: SVD")
st.sidebar.write("Dataset: Netflix Prize Dataset")


st.title("🎬 Netflix Recommendation System")
st.markdown("### Personalized Movie Recommendations using Machine Learning")
st.write("---") 

st.subheader("Enter User Details")

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    step=1
)

num_movies = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=20,
    value=10
)

recommend_button = st.button("Recommend Movies")

if recommend_button:
    st.success(f"Top {num_movies} Movie Recommendations for User {user_id}")

    with st.spinner("Finding the best movies for you... 🎬"):
        recommendations = recommend_movies(user_id, num_movies)

    for i, (title, rating) in enumerate(recommendations, start=1):
        st.markdown(
            f"### {i}. 🎬 {title}\n"
            f"⭐ **Predicted Rating:** {rating}"
        )

    st.write("---")
    st.info(f"Displayed {len(recommendations)} recommendations.")
    st.write("---")
    st.caption("Developed by Vinodini | Netflix Recommendation System using Machine Learning")
    st.sidebar.write("---")
    st.sidebar.info(
    "This project recommends personalized Netflix movies using the SVD collaborative filtering algorithm."
)
 