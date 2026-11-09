"""Synthetic loan data for a reproducible educational demo."""
import numpy as np
import pandas as pd

FEATURES = ["bureau_score", "monthly_income", "loan_amount", "loan_to_income",
            "existing_emi_ratio", "employment_years", "past_due_count"]

def make_dataset(n: int = 2500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    bureau = np.clip(rng.normal(710, 65, n), 450, 850)
    income = np.clip(rng.lognormal(10.8, 0.55, n), 12000, 300000)
    loan = np.clip(rng.lognormal(12.0, 0.55, n), 25000, 900000)
    lti = loan / (income * 12)
    emi_ratio = np.clip(rng.beta(2, 7, n) * 1.4, 0, 1.2)
    employment = np.clip(rng.exponential(4, n), 0, 35)
    past_due = np.clip(rng.poisson(0.35, n), 0, 5)
    logit = (-2.4 - 0.010 * (bureau - 700) + 1.15 * lti
             + 2.0 * emi_ratio - 0.055 * employment + 0.62 * past_due)
    probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, probability)
    return pd.DataFrame({
        "bureau_score": bureau.round().astype(int),
        "monthly_income": income.round(0).astype(int),
        "loan_amount": loan.round(0).astype(int),
        "loan_to_income": lti.round(3),
        "existing_emi_ratio": emi_ratio.round(3),
        "employment_years": employment.round(1),
        "past_due_count": past_due.astype(int),
        "bad_90dpd": target,
    })
