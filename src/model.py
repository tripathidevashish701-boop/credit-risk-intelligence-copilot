"""Model training and transparent coefficient-based explanations."""
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .data import FEATURES, make_dataset

@dataclass
class RiskModel:
    pipeline: Pipeline
    metrics: dict
    feature_names: list

def train_model(random_state: int = 42) -> RiskModel:
    df = make_dataset(random_state=random_state)
    X, y = df[FEATURES], df["bad_90dpd"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    prep = ColumnTransformer([("scale", StandardScaler(), FEATURES)])
    pipe = Pipeline([
        ("preprocess", prep),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "test_rows": int(len(y_test)),
    }
    return RiskModel(pipe, metrics, FEATURES)

def score_applicant(model: RiskModel, applicant: dict) -> dict:
    row = pd.DataFrame([{k: applicant[k] for k in FEATURES}])
    probability = float(model.pipeline.predict_proba(row)[0, 1])
    scaler = model.pipeline.named_steps["preprocess"].named_transformers_["scale"]
    classifier = model.pipeline.named_steps["classifier"]
    scaled = scaler.transform(row)[0]
    contributions = scaled * classifier.coef_[0]
    explanation = sorted(
        [{"feature": name, "contribution": float(value)}
         for name, value in zip(FEATURES, contributions)],
        key=lambda x: abs(x["contribution"]), reverse=True
    )
    return {
        "risk_score": probability,
        "risk_band": "Higher (demo)" if probability >= 0.5 else "Lower (demo)",
        "top_drivers": explanation[:4],
        "disclaimer": "Illustrative synthetic-data output; not a real credit decision."
    }
