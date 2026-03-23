import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# 1) Load dataset
df = pd.read_csv("heart.csv")

# 2) Split input and output
X = df.drop("target", axis=1)
y = df["target"]

# 3) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4) Pipeline (Scaler + Model)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=400,
        random_state=42
    ))
])

# 5) Train model
model.fit(X_train, y_train)

# 6) Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("✅ Training complete")
print("Accuracy:", round(acc * 100, 2), "%")

# 7) Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Saved model.pkl successfully")
