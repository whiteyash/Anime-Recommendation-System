import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load dataset from CSV
# -------------------------------
def load_dataset_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if "title_en" not in df.columns or "genre" not in df.columns or "title_jp" not in df.columns:
            raise ValueError("CSV must have 'title_en', 'title_jp' and 'genre' columns.")
        return df
    except Exception as e:
        raise IOError(f"Error loading dataset: {e}")

# -------------------------------
# Anime Recommender Class
# -------------------------------
class AnimeRecommender:
    def __init__(self, dataframe):
        self.df = dataframe
        cv = CountVectorizer()
        self.count_matrix = cv.fit_transform(self.df["genre"])
        self.cosine_sim = cosine_similarity(self.count_matrix)

    def recommend(self, title, top_n=5):
        if title not in self.df["title_en"].values:
            return [f"❌ Anime '{title}' not found in dataset."]
        
        idx = self.df.index[self.df["title_en"] == title][0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        anime_indices = [i[0] for i in sim_scores]
        
        return self.df["title_en"].iloc[anime_indices].tolist()
