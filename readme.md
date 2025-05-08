
# 🎬 Chayan - Movie Recommender System

Chayan is a personalized movie recommendation system built using Python (Flask) and styled with Tailwind CSS. It allows users to discover movies similar to their preferences using NLP-based cosine similarity on movie overviews.

---

## 🚀 Features

- 🔍 Search and get recommendations based on movie descriptions
- 🔐 User authentication (login system)
- 📊 Movie data visualization (optional future expansion)
- 🧠 NLP + cosine similarity-based recommendation engine
- 🎨 Tailwind CSS-powered responsive UI

---

## 🛠️ Tech Stack

- **Backend:** Python Flask
- **Frontend:** HTML, Tailwind CSS, JavaScript
- **Database:** MySQL (via `flask-mysqldb`)
- **Authentication:** Bcrypt password hashing & Flask session
- **Dataset:** `data/movies.csv` with movie metadata & overviews
- **Recommendation Engine:** Cosine similarity on TF-IDF vectors

---

## 📁 Project Structure

```

chayan/
│
├── app.py                    # Main Flask app
├── utils/
│   └── recommender.py        # Recommendation logic using cosine similarity
├── data/
│   └── movies.csv            # Movie dataset with overviews
├── templates/
│   └── index.html            # Homepage
│   └── login.html            # Login page
├── static/                   # Tailwind CSS and other static assets
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

````

---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/CodeByAmrit/chayan-movie-recommender.git
cd chayan-movie-recommender
````

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup your `.env` or configure MySQL in `app.py`

Edit `app.py` to configure:

```python
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'your_username'
app.config['MYSQL_PASSWORD'] = 'your_password'
app.config['MYSQL_DB'] = 'your_database'
```

### 5. Run the app

```bash
python app.py
```

App will run at: `http://127.0.0.1:5000`

---

## 🔑 Default Routes

* `/` – Homepage
* `/login` – User login
* `/recommend` – Get movie recommendations (based on input)

---

## 🔒 Security Notes

* Passwords are hashed using `bcrypt`.
* Uses Flask sessions for managing user authentication.
* Avoid exposing raw credentials in `app.py` (use `.env` or environment variables in production).

---

## 📝 To-Do / Future Enhancements

* ✅ Add user registration
* ✅ Improve UI with Tailwind components
* ⏳ Add rating and like system
* ⏳ Include genres & poster previews in results
* ⏳ Dockerize the app for easy deployment

---

## 📄 License

MIT License © 2025 [CodeByAmrit](https://github.com/codebyamrit)

---

## 🙌 Contributing

Pull requests are welcome! If you'd like to improve the UI, refactor the backend, or add new features—feel free to fork and contribute.

```

---

Would you like me to create a matching `.gitignore` and `requirements.txt` file now?
```
