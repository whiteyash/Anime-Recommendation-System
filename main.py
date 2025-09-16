from anime_recommender import load_dataset_from_csv, AnimeRecommender

if __name__ == "__main__":
    # Load dataset
    df = load_dataset_from_csv("anime_dataset.csv")
    recommender = AnimeRecommender(df)

    print("🎌 Anime Recommendation System 🎌")
    anime_name = input("Enter an Anime Title (English): ")

    recommendations = recommender.recommend(anime_name, top_n=5)
    print("\nTop Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")
