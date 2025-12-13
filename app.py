import os
import pickle
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast
from dotenv import load_dotenv
load_dotenv()

# ================= CONFIG =================
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://via.placeholder.com/300x450?text=No+Image"

# ================= LOAD DATA =================


@st.cache_data
def load_artifacts():
    movies = pd.read_csv("Data/tmdb_5000_movies.csv")
    credits = pd.read_csv("Data/tmdb_5000_credits.csv")

    # Merge movies + credits
    movies = movies.merge(credits, on="title")

    # Helper functions
    def parse(obj):
        return [i["name"] for i in ast.literal_eval(obj)]

    def parse_cast(obj):
        return [i["name"] for i in ast.literal_eval(obj)[:3]]

    def parse_director(obj):
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                return [i["name"]]
        return []

    # Apply parsing
    movies["genres"] = movies["genres"].apply(parse)
    movies["keywords"] = movies["keywords"].apply(parse)
    movies["cast"] = movies["cast"].apply(parse_cast)
    movies["crew"] = movies["crew"].apply(parse_director)

    # Create tags
    movies["tags"] = (
        movies["overview"].fillna("") + " " +
        movies["genres"].apply(lambda x: " ".join(x)) + " " +
        movies["keywords"].apply(lambda x: " ".join(x)) + " " +
        movies["cast"].apply(lambda x: " ".join(x)) + " " +
        movies["crew"].apply(lambda x: " ".join(x))
    )

    movies = movies[["id", "title", "tags"]]
    movies["tags"] = movies["tags"].str.lower()

    # Vectorization
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    return movies, similarity


# ================= TMDB POSTER =================
def fetch_poster(movie_id):
    if not movie_id or not TMDB_API_KEY:
        return PLACEHOLDER
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            poster = r.json().get("poster_path")
            if poster:
                return TMDB_IMG_BASE + poster
    except:
        pass
    return PLACEHOLDER


# ================= RECOMMENDATION =================
def recommend(movie: str, movies, similarity, top_k: int = 5):
    try:
        idx = movies[movies["title"] == movie].index[0]
    except Exception:
        return [], []

    distances = sorted(
        list(enumerate(similarity[idx])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_names = []
    recommended_posters = []

    for i in distances[1: top_k + 1]:
        movie_id = movies.iloc[i[0]]["id"]   # ✅ FIX HERE
        recommended_names.append(movies.iloc[i[0]]["title"])
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_names, recommended_posters



# ================= MAIN APP =================
def main():
    st.set_page_config("Movie Recommender", layout="wide")
    st.title("🎬 Movies Recommendation System")

    # ---------- CSS for horizontal row ----------
    st.markdown("""
    <style>
    .scroll-row {
        display: flex;
        gap: 40px;
        overflow-x: auto;
        padding-bottom: 30px;
        width: 100%;
    }

    .movie-card {
        min-width: 700px;     /* base size */
        width: 100%;          /* allow expansion */
        display: flex;
        gap: 20px;
        flex-shrink: 0;
    }


    .movie-poster {
        width: 180px;
        border-radius: 12px;
    }

    .movie-info {
        flex: 1;
    }

    .movie-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 6px;
    }

    .movie-meta {
        color: #b5b5b5;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }

    .movie-overview {
        font-size: 0.95rem;
        line-height: 1.6em;
        color: #e5e5e5;
        white-space: normal;
    }

    </style>
    """, unsafe_allow_html=True)



    movies, similarity = load_artifacts()
    movies_csv = pd.read_csv("Data/tmdb_5000_movies.csv")



    # ---------- SELECT MOVIE ----------
    selected_movie = st.selectbox(
        "Type or select a movie to get a recommendation",
        movies["title"].values
    )

    # ---------- HELPER ----------
    def get_local_details(movie_id):
        import ast, pandas as pd
        row = movies_csv[movies_csv["id"] == int(movie_id)]
        if row.empty:
            return {}
        row = row.iloc[0]
        details = {
            "overview": row.get("overview", ""),
            "release_date": row.get("release_date", ""),
            "vote_average": row.get("vote_average", ""),
            "vote_count": row.get("vote_count", ""),
            "runtime": row.get("runtime", ""),
            "tagline": row.get("tagline", ""),
            "genres": []
        }
        try:
            if pd.notna(row.get("genres")):
                details["genres"] = [
                    g["name"] for g in ast.literal_eval(row["genres"])
                ]
        except:
            pass
        return details

    # ---------- BUTTON ----------
    if st.button("Show recommendation"):
        names, posters = recommend(selected_movie, movies, similarity)

        # ===== STEP 2: SELECTED MOVIE DETAILS =====
        sel_idx = movies[movies["title"] == selected_movie].index[0]
        sel_id = movies.iloc[sel_idx]["movie_id"]
        sel_details = get_local_details(sel_id)

        st.subheader(selected_movie)
        st.write(f"**Release date:** {sel_details.get('release_date','-')}")
        st.write(f"**Rating:** {sel_details.get('vote_average','-')} ({sel_details.get('vote_count','-')} votes)")
        st.write(f"**Runtime:** {sel_details.get('runtime','-')} mins")
        st.write(f"**Genres:** {', '.join(sel_details.get('genres',[]))}")
        if sel_details.get("tagline"):
            st.write(f"**Tagline:** {sel_details['tagline']}")
        if sel_details.get("overview"):
            st.write(sel_details["overview"])

        # ===== STEP 3: DIVIDER + HEADING =====
        st.markdown("---")
        st.subheader("Recommended Movies")

        # ===== STEP 4: HORIZONTAL NETFLIX ROW =====
        st.markdown("<div class='scroll-row'>", unsafe_allow_html=True)

        for name, poster in zip(names, posters):
            try:
                r = movies[movies["title"] == name].index[0]
                mid = movies.iloc[r]["movie_id"]
            except:
                mid = None

            d = get_local_details(mid) if mid else {}

            overview_text = d.get("overview") or "No overview available"

            st.markdown(
                f"""
                <div class="movie-card">
                    <img src="{poster}" class="movie-poster" />
                    <div class="movie-info">
                        <div class="movie-title">{name}</div>
                        <div class="movie-meta">
                            {d.get('release_date','-')} — ⭐ {d.get('vote_average','-')}
                        </div>
                        <div class="movie-overview">
                            {overview_text}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ================= RUN =================
if __name__ == "__main__":
    main()
