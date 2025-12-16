from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from career_recommendation import recommend_careers

app = Flask(__name__)
CORS(app)  # Allow React (localhost:3000)

# Load trained model & label encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Career Recommendation API is running"}), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        features = [
            float(data.get("tenth", 0)),
            float(data.get("twelfth", 0)),
            float(data.get("ug_cgpa", 0)),
            float(data.get("math", 0)),
            float(data.get("programming", 0)),
            float(data.get("communication", 0)),
            float(data.get("creativity", 0)),
            float(data.get("interest_it", 0)),
            float(data.get("interest_medical", 0)),
            float(data.get("interest_arts", 0)),
        ]

        # Predict primary career
        pred_encoded = model.predict([features])[0]
        primary_career = label_encoder.inverse_transform([pred_encoded])[0]

        # Recommend top careers
        user_vector = np.array(features[3:])  # last 7 values (skills + interests)
        recommendations_df = recommend_careers(user_vector, top_n=5)
        recommendations = recommendations_df[["Career", "Score"]].to_dict(orient="records")

        return jsonify({
            "primary_career": primary_career,
            "recommendations": recommendations
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Invalid input or server error"}), 400

if __name__ == "__main__":
    app.run(debug=True)
