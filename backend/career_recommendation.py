import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

careers_df = pd.read_csv("careers.csv")

FEATURES = [
    "Math",
    "Programming",
    "Communication",
    "Creativity",
    "Interest_IT",
    "Interest_Medical",
    "Interest_Arts",
]

def recommend_careers(user_vector, top_n=5):
    similarities = []

    for _, row in careers_df.iterrows():
        career_vector = row[FEATURES].values.astype(float)
        score = cosine_similarity([user_vector], [career_vector])[0][0] * 100
        similarities.append(score)

    temp = careers_df.copy()
    temp["Score"] = similarities
    temp = temp.sort_values(by="Score", ascending=False).head(top_n)
    temp["Score"] = temp["Score"].round(2)
    return temp
