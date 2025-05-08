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

# Load environment variables
load_dotenv()

def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("SECRET_KEY")

    # MySQL Configuration for Aiven
    app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
    app.config['MYSQL_PORT'] = int(os.getenv('MYSQL_PORT', 3306))
    app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
    app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
    app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')
    app.config['MYSQL_SSL_CA'] = os.getenv('MYSQL_SSL_CA')
    app.config['MYSQL_SSL_VERIFY_IDENTITY'] = os.getenv('MYSQL_SSL_VERIFY_IDENTITY', 'true').lower() == 'true'
    app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

    # Initialize MySQL
    mysql = MySQL(app)

    # TMDB API Configuration
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
                # Load movies from database
                cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cur.execute("SELECT * FROM movies")
                movies = cur.fetchall()
                cur.close()

                self.movies_df = pd.DataFrame(movies)

                # Create TF-IDF matrix for content-based filtering
                if not self.movies_df.empty and "overview" in self.movies_df:
                    tfidf = TfidfVectorizer(stop_words="english")
                    self.tfidf_matrix = tfidf.fit_transform(
                        self.movies_df["overview"].fillna("")
                    )

                    # Load ratings for collaborative filtering
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

                        # Create KNN model for collaborative filtering
                        features = pd.get_dummies(
                            self.movies_df["genres"]
                            .str.split(",", expand=True)
                            .stack()
                            .str.strip()
                        )
                        features = features.groupby(level=0).sum()
                        self.knn_model = NearestNeighbors(n_neighbors=5, algorithm="auto")
                        self.knn_model.fit(features)

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

    # Initialize recommender within app context
    with app.app_context():
        recommender = RecommendationEngine()

    # Helper Functions
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
                INSERT INTO movies (tmdb_id, title, overview, genres, release_date, poster_path, rating)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    title = VALUES(title),
                    overview = VALUES(overview),
                    genres = VALUES(genres),
                    release_date = VALUES(release_date),
                    poster_path = VALUES(poster_path),
                    rating = VALUES(rating)
            """,
                (
                    tmdb_id,
                    movie_data["title"],
                    movie_data["overview"],
                    ",".join([g["name"] for g in movie_data["genres"]]),
                    movie_data["release_date"],
                    movie_data["poster_path"],
                    movie_data["vote_average"],
                ),
            )
            mysql.connection.commit()
            movie_id = cur.lastrowid
            cur.close()

            # Refresh recommendation engine
            recommender.load_data()

        return movie_id

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
                    cur.close()
                return redirect(url_for("login"))
            except Exception as e:
                return render_template("register.html", error="Username already exists")

        return render_template("register.html")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

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

            # Get recommendations
            recommendations = recommender.hybrid_recommendations(session["user_id"], movie_id)

            # Check if user has rated this movie
            cur.execute(
                """
                SELECT rating FROM user_preferences 
                WHERE user_id = %s AND movie_id = %s
            """,
                (session["user_id"], movie_id),
            )
            user_rating = cur.fetchone()
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

            # Refresh recommendation engine
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

        # If not enough local results, search TMDB
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

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)