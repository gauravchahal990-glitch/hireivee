# services/predictor.py

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURES = [
    "age", "degree_code", "branch_code", "cgpa", "internships",
    "projects", "coding", "aptitude", "communication",
    "soft_skills", "backlogs", "certifications"
]

DEGREE_CODES = {
    "B.Tech": 0, "B.E.": 1, "BCA": 2, "MCA": 3,
    "M.Tech": 4, "B.Sc": 5, "Other": 6
}

BRANCH_CODES = {
    "Computer Science": 0, "Information Technology": 1,
    "Electronics": 2, "Mechanical": 3, "Civil": 4,
    "Electrical": 5, "Other": 6
}


class PlacementPredictor:
    """Random Forest baseline trained on synthetic demo data."""

    def __init__(self):
        self.model = None
        self.model_name = "Random Forest"
        self.validation_accuracy = None

    def _make_demo_data(self, n=4500, seed=42):
        rng = np.random.default_rng(seed)

        df = pd.DataFrame({
            "age": rng.integers(20, 27, n),
            "degree_code": rng.integers(0, 7, n),
            "branch_code": rng.integers(0, 7, n),
            "cgpa": rng.uniform(5.0, 10.0, n),
            "internships": rng.integers(0, 4, n),
            "projects": rng.integers(0, 7, n),
            "coding": rng.uniform(1, 10, n),
            "aptitude": rng.uniform(1, 10, n),
            "communication": rng.uniform(1, 10, n),
            "soft_skills": rng.uniform(1, 10, n),
            "backlogs": rng.integers(0, 5, n),
            "certifications": rng.integers(0, 8, n)
        })

        score = (
            df.cgpa * 0.9
            + df.internships * 0.8
            + df.projects * 0.35
            + df.coding * 0.55
            + df.aptitude * 0.35
            + df.communication * 0.3
            + df.soft_skills * 0.2
            + df.certifications * 0.12
            - df.backlogs * 1.2
        )

        probability = 1 / (1 + np.exp(-(score - 12.5) / 3.2))
        y = rng.binomial(1, probability)

        return df[FEATURES], y

    def load_or_train(self, model_path):
        model_path = Path(model_path)

        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                return
            except Exception:
                pass

        X, y = self._make_demo_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced"
        )

        self.model.fit(X_train, y_train)

        self.validation_accuracy = round(
            accuracy_score(y_test, self.model.predict(X_test)) * 100, 1
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)

    def predict_probability(self, profile):
        values = dict(profile)

        values["degree_code"] = DEGREE_CODES.get(
            str(profile.get("degree", "Other")), 6
        )

        values["branch_code"] = BRANCH_CODES.get(
            str(profile.get("branch", "Other")), 6
        )

        row = [[float(values.get(key, 0)) for key in FEATURES]]

        return round(
            float(self.model.predict_proba(row)[0][1]) * 100, 2
        )# hireivee
