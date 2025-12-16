import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# For reproducibility
np.random.seed(42)

# ---------- 1) GENERATE SYNTHETIC DATASET ----------
n = 1200

careers = [
    "Software Engineer",
    "Data Scientist",
    "Graphic Designer",
    "Doctor",
    "Teacher",
    "Civil Engineer",
    "Mechanical Engineer",
    "Accountant"
]

rows = []
for i in range(n):
    tenth = np.random.randint(55, 100)
    twelfth = np.random.randint(50, 100)
    ug_cgpa = round(np.random.uniform(6.0, 9.8), 2)

    math = np.random.randint(1, 11)
    programming = np.random.randint(1, 11)
    communication = np.random.randint(1, 11)
    creativity = np.random.randint(1, 11)
    it_interest = np.random.randint(1, 11)
    medical_interest = np.random.randint(1, 11)
    arts_interest = np.random.randint(1, 11)

    # Simple rule-based logic to assign a realistic career label
    if programming >= 7 and it_interest >= 7:
        career = "Software Engineer"
    elif math >= 8 and programming >= 6 and it_interest >= 6:
        career = "Data Scientist"
    elif medical_interest >= 8 and math >= 6:
        career = "Doctor"
    elif creativity >= 8 and arts_interest >= 7:
        career = "Graphic Designer"
    elif communication >= 7 and arts_interest >= 5:
        career = "Teacher"
    elif math >= 7 and it_interest <= 6:
        career = "Civil Engineer"
    elif math >= 6 and creativity >= 5 and medical_interest <= 5:
        career = "Mechanical Engineer"
    else:
        career = "Accountant"

    rows.append({
        "10th": tenth,
        "12th": twelfth,
        "UG_CGPA": ug_cgpa,
        "Math": math,
        "Programming": programming,
        "Communication": communication,
        "Creativity": creativity,
        "Interest_IT": it_interest,
        "Interest_Medical": medical_interest,
        "Interest_Arts": arts_interest,
        "Career": career
    })

df = pd.DataFrame(rows)

# Save the dataset to CSV (for your reference)
df.to_csv("students_dataset_1200.csv", index=False)
print("✅ Generated students_dataset_1200.csv with shape:", df.shape)

# ---------- 2) TRAIN ML MODEL ON THIS DATA ----------
# Encode target labels (Career)
le = LabelEncoder()
df["Career"] = le.fit_transform(df["Career"])

# Features and target
X = df.drop("Career", axis=1)
y = df["Career"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Save model & label encoder
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

acc = model.score(X_test, y_test)
print(f"✅ Model trained successfully. Test accuracy: {acc:.2f}")
