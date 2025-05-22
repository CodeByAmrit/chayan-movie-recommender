import json
import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_mysqldb import MySQL
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import requests
from dotenv import load_dotenv
import bcrypt
from datetime import datetime
import MySQLdb
from flask import Flask, request, jsonify
import google.generativeai as genai
import os

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("SECRET_KEY")

    app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
    app.config['MYSQL_PORT'] = int(os.getenv('MYSQL_PORT', 3306))
    app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
    app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
    app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')
    app.config['MYSQL_SSL_CA'] = os.getenv('MYSQL_SSL_CA')
    app.config['MYSQL_SSL_VERIFY_IDENTITY'] = os.getenv('MYSQL_SSL_VERIFY_IDENTITY', 'true').lower() == 'true'
    app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

    mysql = MySQL(app)

    TMDB_API_KEY = os.getenv("TMDB_API_KEY")
    TMDB_BASE_URL = "https://api.themoviedb.org/3"

    class RecommendationEngine:
        def __init__(self):
            self.movies_df = None
            self.tfidf_matrix = None
            self.knn_model = None
            self.load_data()

        def load_data(self):
            with app.app_context():                
                cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cur.execute("SELECT * FROM movies")
                movies = cur.fetchall()
                cur.close()

                self.movies_df = pd.DataFrame(movies)

                if not self.movies_df.empty and "overview" in self.movies_df:
                    tfidf = TfidfVectorizer(stop_words="english")
                    self.tfidf_matrix = tfidf.fit_transform(
                        self.movies_df["overview"].fillna("")
                    )

                    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                    cur.execute("SELECT user_id, movie_id, rating FROM user_preferences")
                    ratings = cur.fetchall()
                    cur.close()

                    if ratings:
                        ratings_df = pd.DataFrame(ratings)
                        movie_ratings = ratings_df.groupby("movie_id")["rating"].mean().reset_index()
                        self.movies_df = self.movies_df.merge(
                            movie_ratings, left_on="id", right_on="movie_id", how="left"
                        )
                        self.movies_df["rating"] = self.movies_df["rating_y"].fillna(
                            self.movies_df["rating_x"]
                        )
                        self.movies_df.drop(
                            ["rating_x", "rating_y", "movie_id"], axis=1, inplace=True
                        )

                        features = pd.get_dummies(
                            self.movies_df["genres"]
                            .str.split(",", expand=True)
                            .stack()
                            .str.strip()
                        )
                        features = features.groupby(level=0).sum()
                        self.knn_model = NearestNeighbors(n_neighbors=5, algorithm="auto")
                        self.knn_model.fit(features)

        def hybrid_recommendations(self, user_id, movie_id=None, n=5):
            # Get user preferences
            with app.app_context():
                cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

                # Get genre preferences
                cur.execute("""
                    SELECT g.name as genre 
                    FROM user_genre_preferences ugp
                    JOIN genres g ON ugp.genre_id = g.id
                    WHERE ugp.user_id = %s
                """, (user_id,))
                user_genres = [row['genre'] for row in cur.fetchall()]

                # Get language preferences
                cur.execute("""
                    SELECT l.name as language 
                    FROM user_language_preferences ulp
                    JOIN languages l ON ulp.language_id = l.id
                    WHERE ulp.user_id = %s
                """, (user_id,))
                user_languages = [row['language'] for row in cur.fetchall()]

                # Get decade preference
                cur.execute("""
                    SELECT decade 
                    FROM user_decade_preferences 
                    WHERE user_id = %s
                """, (user_id,))
                decade_row = cur.fetchone()
                decade = decade_row['decade'] if decade_row else None

                cur.close()

            # Start with all movies
            filtered_movies = self.movies_df.copy()

            # Apply genre filter if any
            if user_genres:
                filtered_movies = filtered_movies[filtered_movies['genres'].apply(lambda x: any(genre in x for genre in user_genres))]

            # Apply language filter if any
            if user_languages and 'original_language' in filtered_movies.columns:
                filtered_movies = filtered_movies[filtered_movies['original_language'].isin([lang.lower()[:2] for lang in user_languages])]

            # Apply decade filter if any
            if decade and 'release_date' in filtered_movies.columns:
                if decade == '2020s':
                    filtered_movies = filtered_movies[
                        filtered_movies['release_date'] >= '2020-01-01'
                        ]
                elif decade == '2010s':
                    filtered_movies = filtered_movies[
                        (filtered_movies['release_date'] >= '2010-01-01') &
                        (filtered_movies['release_date'] < '2020-01-01')
                        ]
                elif decade == '2000s':
                    filtered_movies = filtered_movies[
                        (filtered_movies['release_date'] >= '2000-01-01') &
                        (filtered_movies['release_date'] < '2010-01-01')
                        ]
                # Add more decades as needed

            # If no movies left after filtering, use all movies
            if filtered_movies.empty:
                filtered_movies = self.movies_df.copy()

            # Rest of the recommendation logic remains the same
            cb_recs = []
            cf_recs = []

            if movie_id and movie_id in filtered_movies['id'].values:
                cb_recs = self.content_based_recommendations(movie_id, n)

            if user_id:
                cf_recs = self.collaborative_filtering_recommendations(user_id, n)

            # Combine and deduplicate recommendations
            combined = cb_recs + cf_recs
            seen = set()
            unique_recs = []

            for rec in combined:
                if rec["id"] not in seen:
                    seen.add(rec["id"])
                    unique_recs.append(rec)
                    if len(unique_recs) >= n:
                        break

            return unique_recs
        def content_based_recommendations(self, movie_id, n=5):
            if self.tfidf_matrix is None:
                return []

            idx = self.movies_df[self.movies_df["id"] == movie_id].index[0]
            sim_scores = list(
                enumerate(cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0])
            )
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1 : n + 1]
            movie_indices = [i[0] for i in sim_scores]
            return self.movies_df.iloc[movie_indices].to_dict("records")

        def collaborative_filtering_recommendations(self, user_id, n=5):
            if self.knn_model is None:
                return []

            with app.app_context():
                cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cur.execute(
                    """
                    SELECT movie_id, rating 
                    FROM user_preferences 
                    WHERE user_id = %s 
                    ORDER BY rating DESC 
                    LIMIT 1
                """,
                    (user_id,),
                )
                top_rated = cur.fetchone()
                cur.close()

                if not top_rated:
                    return []

                movie_id = top_rated["movie_id"]
                idx = self.movies_df[self.movies_df["id"] == movie_id].index[0]

                features = pd.get_dummies(
                    self.movies_df["genres"].str.split(",", expand=True).stack().str.strip()
                )
                features = features.groupby(level=0).sum()

                distances, indices = self.knn_model.kneighbors(
                    features.iloc[idx : idx + 1], n_neighbors=n + 1
                )
                return self.movies_df.iloc[indices[0][1:]].to_dict("records")

        def hybrid_recommendations(self, user_id, movie_id=None, n=5):
            cb_recs = []
            cf_recs = []

            if movie_id:
                cb_recs = self.content_based_recommendations(movie_id, n)

            if user_id:
                cf_recs = self.collaborative_filtering_recommendations(user_id, n)

            combined = cb_recs + cf_recs
            seen = set()
            unique_recs = []

            for rec in combined:
                if rec["id"] not in seen:
                    seen.add(rec["id"])
                    unique_recs.append(rec)
                    if len(unique_recs) >= n:
                        break

            return unique_recs

    with app.app_context():
        recommender = RecommendationEngine()

    def fetch_movie_from_tmdb(movie_id):
        url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None

    def search_movies_on_tmdb(query):
        url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={query}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []

    def add_movie_to_db(tmdb_id):
        movie_data = fetch_movie_from_tmdb(tmdb_id)
        if not movie_data:
            return False

        with app.app_context():
            cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cur.execute(
                """
                INSERT INTO movies (
                    tmdb_id, title, overview, genres, release_date, 
                    poster_path, rating, original_language
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    title = VALUES(title),
                    overview = VALUES(overview),
                    genres = VALUES(genres),
                    release_date = VALUES(release_date),
                    poster_path = VALUES(poster_path),
                    rating = VALUES(rating),
                    original_language = VALUES(original_language)
                """,
                (
                    tmdb_id,
                    movie_data["title"],
                    movie_data["overview"],
                    ",".join([g["name"] for g in movie_data["genres"]]),
                    movie_data["release_date"],
                    movie_data["poster_path"],
                    movie_data["vote_average"],
                    movie_data["original_language"],
                ),
            )
            mysql.connection.commit()
            movie_id = cur.lastrowid
            cur.close()

            # Refresh recommendation engine
            recommender.load_data()

        return movie_id

    @app.route("/privacy")
    def privacy():
        return render_template("privacy.html")

    @app.route("/terms")
    def terms():

        return render_template("terms.html")

    # Routes
    @app.route("/")
    def home():
        if "user_id" not in session:
            return redirect(url_for("login"))

        with app.app_context():
            cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cur.execute("SELECT * FROM movies ORDER BY rating DESC LIMIT 12")
            popular_movies = cur.fetchall()
            cur.close()

        return render_template("index.html", movies=popular_movies)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form["username"]
            password = request.form["password"].encode("utf-8")

            with app.app_context():
                cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cur.execute("SELECT * FROM users WHERE username = %s", (username,))
                user = cur.fetchone()
                cur.close()

                if user:
                    stored_password = user["password"]
                    if isinstance(stored_password, str):
                        stored_password = stored_password.encode("utf-8")

                    if bcrypt.checkpw(password, stored_password):
                        session["user_id"] = user["id"]
                        session["username"] = user["username"]
                        return redirect(url_for("home"))

            return render_template("login.html", error="Invalid credentials")

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route('/movie/<int:movie_id>/providers')
    def movie_providers(movie_id):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
        headers = {"Authorization": f"Bearer {TMDB_API_KEY}"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch providers"}), 500

        data = response.json().get("results", {})

        all_countries = {}

        for country_code, country_data in data.items():
            all_countries[country_code] = {
                "link": country_data.get("link"),
                "flatrate": [
                    {
                        "provider_name": p.get("provider_name"),
                        "logo_path": f"https://image.tmdb.org/t/p/w200{p.get('logo_path')}"
                    }
                    for p in country_data.get("flatrate", [])
                ],
                "rent": [
                    {
                        "provider_name": p.get("provider_name"),
                        "logo_path": f"https://image.tmdb.org/t/p/w200{p.get('logo_path')}"
                    }
                    for p in country_data.get("rent", [])
                ],
                "buy": [
                    {
                        "provider_name": p.get("provider_name"),
                        "logo_path": f"https://image.tmdb.org/t/p/w200{p.get('logo_path')}"
                    }
                    for p in country_data.get("buy", [])
                ],
                "free": [
                    {
                        "provider_name": p.get("provider_name"),
                        "logo_path": f"https://image.tmdb.org/t/p/w200{p.get('logo_path')}"
                    }
                    for p in country_data.get("free", [])
                ]
            }

        return jsonify(all_countries)

    @app.route('/movie/<int:movie_id>/providers/html')
    def movie_providers_html(movie_id):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
        headers = {"Authorization": f"Bearer {TMDB_API_KEY}"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return "Error fetching providers", 500

        data = response.json().get("results", {})
        all_countries = {}

        for country_code, country_data in data.items():
            all_countries[country_code] = {
                "link": country_data.get("link"),
                "flatrate": [
                    {
                        "provider_name": p.get("provider_name"),
                        "logo_path": f"https://image.tmdb.org/t/p/w200{p.get('logo_path')}"
                    }
                    for p in country_data.get("flatrate", [])
                ],
                "rent": [
                    {
                        "provider_name": p.get("provider_name"),
                        "logo_path": f"https://image.tmdb.org/t/p/w200{p.get('logo_path')}"
                    }
                    for p in country_data.get("rent", [])
                ],
                "buy": [
                    {
                        "provider_name": p.get("provider_name"),
                        "logo_path": f"https://image.tmdb.org/t/p/w200{p.get('logo_path')}"
                    }
                    for p in country_data.get("buy", [])
                ],
                "free": [
                    {
                        "provider_name": p.get("provider_name"),
                        "logo_path": f"https://image.tmdb.org/t/p/w200{p.get('logo_path')}"
                    }
                    for p in country_data.get("free", [])
                ]
            }

        return render_template("providers.html", providers=all_countries)

    @app.route("/movie/<int:movie_id>")
    def movie_detail(movie_id):
        if "user_id" not in session:
            return redirect(url_for("login"))

        with app.app_context():
            cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cur.execute("SELECT * FROM movies WHERE id = %s", (movie_id,))
            movie = cur.fetchone()

            if not movie:
                cur.close()
                return redirect(url_for("home"))

            # TMDB Watch Providers API
            tmdb_api_key = os.getenv("TMDB_API_KEY")
            provider_url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
            response = requests.get(provider_url, params={"api_key": tmdb_api_key})

            platforms = []
            if response.status_code == 200:
                provider_data = response.json().get("results", {}).get("IN")  # 'IN' for India
                if provider_data and "flatrate" in provider_data:
                    for provider in provider_data["flatrate"]:
                        platforms.append({
                            "name": provider["provider_name"],
                            "url": provider_data.get("link", "#"),
                            "logo_path": provider["logo_path"]
                        })

            movie["platforms"] = provider_url
            try:
                movie['platforms'] = json.loads(movie['platforms'])
            except json.JSONDecodeError:
                movie['platforms'] = {}
            print(response.json().get("results", {}).get("IN"))

            # User rating
            cur.execute(
                """
                SELECT rating FROM user_preferences 
                WHERE user_id = %s AND movie_id = %s
            """,
                (session["user_id"], movie_id),
            )
            user_rating = cur.fetchone()

            # Recommendations
            recommendations = recommender.hybrid_recommendations(session["user_id"], movie_id)

            cur.close()

        return render_template(
            "movie_detail.html",
            movie=movie,
            recommendations=recommendations,
            user_rating=user_rating["rating"] if user_rating else None,
        )

    @app.route("/rate_movie", methods=["POST"])
    def rate_movie():
        if "user_id" not in session:
            return jsonify({"status": "error", "message": "Not logged in"}), 401

        movie_id = request.form.get("movie_id")
        rating = request.form.get("rating")

        if not movie_id or not rating:
            return jsonify({"status": "error", "message": "Missing data"}), 400

        with app.app_context():
            cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cur.execute(
                """
                INSERT INTO user_preferences (user_id, movie_id, rating, timestamp)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE rating = VALUES(rating), timestamp = VALUES(timestamp)
            """,
                (session["user_id"], movie_id, rating, datetime.now()),
            )
            mysql.connection.commit()
            cur.close()

            recommender.load_data()

        return jsonify({"status": "success"})

    @app.route("/search")
    def search():
        query = request.args.get("q", "")
        if not query:
            return redirect(url_for("home"))

        with app.app_context():
            cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cur.execute(
                """
                SELECT * FROM movies 
                WHERE title LIKE %s 
                ORDER BY rating DESC 
                LIMIT 20
            """,
                (f"%{query}%",),
            )
            local_results = cur.fetchall()
            cur.close()

        if len(local_results) < 5:
            tmdb_results = search_movies_on_tmdb(query)
            for result in tmdb_results:
                add_movie_to_db(result["id"])

            with app.app_context():
                cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cur.execute(
                    """
                    SELECT * FROM movies 
                    WHERE title LIKE %s 
                    ORDER BY rating DESC 
                    LIMIT 20
                """,
                    (f"%{query}%",),
                )
                local_results = cur.fetchall()
                cur.close()

        return render_template("search.html", query=query, results=local_results)

    @app.route("/recommendations")
    def get_recommendations():
        if "user_id" not in session:
            return jsonify({"status": "error", "message": "Not logged in"}), 401

        recommendations = recommender.hybrid_recommendations(session["user_id"])
        return jsonify({"status": "success", "recommendations": recommendations})

    @app.route('/save_preferences', methods=['POST'])
    def save_preferences():
        if 'user_id' not in session:
            return redirect(url_for('login'))

        genres = request.form.getlist('genres')
        languages = request.form.getlist('languages')
        decade = request.form.get('decade')

        with app.app_context():
            cur = mysql.connection.cursor()

            # Clear existing preferences
            cur.execute("DELETE FROM user_genre_preferences WHERE user_id = %s", (session['user_id'],))
            cur.execute("DELETE FROM user_language_preferences WHERE user_id = %s", (session['user_id'],))
            cur.execute("DELETE FROM user_decade_preferences WHERE user_id = %s", (session['user_id'],))

            # Save new preferences
            for genre_id in genres[:3]:  # Only save up to 3 genres
                cur.execute(
                    "INSERT INTO user_genre_preferences (user_id, genre_id) VALUES (%s, %s)",
                    (session['user_id'], genre_id)
                )

            for language_id in languages:
                cur.execute(
                    "INSERT INTO user_language_preferences (user_id, language_id) VALUES (%s, %s)",
                    (session['user_id'], language_id)
                )

            if decade:
                cur.execute(
                    "INSERT INTO user_decade_preferences (user_id, decade) VALUES (%s, %s)",
                    (session['user_id'], decade)
                )

            mysql.connection.commit()
            cur.close()

        return redirect(url_for('home'))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            username = request.form["username"]
            password = request.form["password"].encode("utf-8")
            hashed = bcrypt.hashpw(password, bcrypt.gensalt())

            try:
                with app.app_context():
                    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                    cur.execute(
                        "INSERT INTO users (username, password) VALUES (%s, %s)",
                        (username, hashed.decode("utf-8")),
                    )
                    mysql.connection.commit()
                    user_id = cur.lastrowid
                    cur.close()

                    session["user_id"] = user_id
                    session["username"] = username
                    return redirect(url_for('preferences'))
            except Exception as e:
                return render_template("register.html", error="Username already exists")

        return render_template("register.html")

    @app.route("/preferences")
    def preferences():
        if 'user_id' not in session:
            return redirect(url_for('login'))

        with app.app_context():
            cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cur.execute("SELECT * FROM genres")
            genres = cur.fetchall()
            cur.execute("SELECT * FROM languages")
            languages = cur.fetchall()
            cur.close()

        return render_template("preferences.html", genres=genres, languages=languages)

    @app.route('/chat', methods=['POST'])
    def chat():
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        try:
            # Create a movie-focused prompt with clear formatting instructions
            prompt = f"""
            You are a helpful movie expert assistant for a movie recommendation platform. 
            The user is asking: "{user_message}"
            
            Please provide a concise, informative response about movies, actors, directors, 
            or anything film-related. Format your response as follows:
            
            - Use **double asterisks** for important names/titles (e.g., **Brie Larson**)
            - Use *single asterisks* for emphasis (e.g., *Captain Marvel*)
            - For lists, use asterisk + space (e.g., * Item 1)
            - Keep responses under 300 words.
            
            Example format for cast information:
            The main cast includes:
            * **Actor Name** as Character Name
            * **Actor Name** as Character Name
            
            For non-movie questions, politely redirect back to movie topics.
            """
            
            response = model.generate_content(prompt)
            
            # Clean up the response if needed
            cleaned_response = response.text.replace('•', '*')  # Convert bullet points if needed
            
            return jsonify({'response': cleaned_response})
        
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            return jsonify({'response': "Sorry, I'm having trouble answering that. Please try again later."})
    
    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
