"""REVIEWED COPY of Ridge_Lasso.py - original file left untouched.
The leave-one-city-out evaluation (k = 0, 1, 2) is UNCHANGED. Changes:
  1. The penalty (alpha) is now chosen with cross-validation split BY CITY inside each training set
     (the original used row-wise splits, which let quarters of the same city sit in both parts).
  2. The alpha grid now reaches lower (1e-3); Lasso kept hitting the bottom of the old grid (0.01).
  3. The chosen alphas are printed.
  4. Added a coefficient comparison against ordinary regression (a checklist item in the starter doc).
  5. Added the interpretation text.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def evaluate_penalized_models(df, features, target):
    cities = df['city'].unique()

    # Dictionaries to store sum of squared errors (SSE) and total sum of squares (SST)
    results = {
        'Ridge': {
            'k0_sq_err': [],
            'k0_sq_tot': [],
            'k1_sq_err': [],
            'k1_sq_tot': [],
            'k2_sq_err': [],
            'k2_sq_tot': [],
        },
        'Lasso': {
            'k0_sq_err': [],
            'k0_sq_tot': [],
            'k1_sq_err': [],
            'k1_sq_tot': [],
            'k2_sq_err': [],
            'k2_sq_tot': [],
        },
    }

    ALPHAS = np.logspace(-3, 3, 100)
    chosen_alpha = {'Ridge': [], 'Lasso': []}

    # Leave-one-city-out cross-validation
    for city in cities:
        # Split train (40 cities) and test (1 left-out city)
        train = df[df['city'] != city]
        test = df[df['city'] == city].sort_values(['year', 'quarter'])

        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target].values

        # Standardize features (crucial for Ridge/Lasso) based on training data
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # Baseline guess: Mean SO2 of the other 40 cities
        baseline_pred = y_train.mean()

        # Penalty chosen by 5-fold cross-validation split BY CITY, using the 40 training cities only
        city_folds = list(GroupKFold(n_splits=5).split(X_train_sc, y_train, groups=train['city']))
        models = {
            'Ridge': RidgeCV(alphas=ALPHAS, cv=city_folds, scoring='neg_mean_squared_error'),
            'Lasso': LassoCV(alphas=ALPHAS, cv=city_folds, max_iter=10000),
        }

        for name, model in models.items():
            # Fit model on the 40 cities
            model.fit(X_train_sc, y_train)
            chosen_alpha[name].append(model.alpha_)

            # Predict the 12 quarters of the held-out city
            preds = model.predict(X_test_sc)

            # --- k = 0: Zero information ---
            results[name]['k0_sq_err'].extend((y_test - preds) ** 2)
            results[name]['k0_sq_tot'].extend((y_test - baseline_pred) ** 2)

            # --- k = 1: Reveal first quarter ---
            miss_k1 = y_test[0] - preds[0]
            preds_k1 = preds[1:] + miss_k1  # Apply offset to remaining 11 quarters
            results[name]['k1_sq_err'].extend((y_test[1:] - preds_k1) ** 2)
            results[name]['k1_sq_tot'].extend((y_test[1:] - baseline_pred) ** 2)

            # --- k = 2: Reveal first two quarters ---
            miss_k2 = np.mean(y_test[:2] - preds[:2])
            preds_k2 = preds[2:] + miss_k2  # Apply offset to remaining 10 quarters
            results[name]['k2_sq_err'].extend((y_test[2:] - preds_k2) ** 2)
            results[name]['k2_sq_tot'].extend((y_test[2:] - baseline_pred) ** 2)

    # 3. Calculate and print pooled metrics
    print(f"{'Model':<10} | {'k':<2} | {'Pooled RMSE':<12} | {'Pooled R²':<10}")
    print("-" * 50)
    for name in ['Ridge', 'Lasso']:
        for k in [0, 1, 2]:
            sq_err = np.sum(results[name][f'k{k}_sq_err'])
            sq_tot = np.sum(results[name][f'k{k}_sq_tot'])
            n = len(results[name][f'k{k}_sq_err'])

            rmse = np.sqrt(sq_err / n)
            r2 = 1 - (sq_err / sq_tot)

            print(f"{name:<10} | {k:<2} | {rmse:<12.2f} | {r2:<10.3f}")

    print("\nChosen alpha over the 41 leave-one-city-out fits (features are standardised):")
    for name, a in chosen_alpha.items():
        print(f"  {name}: median {np.median(a):.3f}, min {np.min(a):.3f}, max {np.max(a):.3f}")
    return chosen_alpha


def compare_coefficients(df, features, target):
    """Fit ordinary regression, Ridge and Lasso on all 41 cities and compare the coefficients (original units)."""
    ALPHAS = np.logspace(-3, 3, 100)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[features])
    folds = list(GroupKFold(n_splits=5).split(Xs, df[target], groups=df['city']))
    ols = LinearRegression().fit(df[features], df[target])
    ridge = RidgeCV(alphas=ALPHAS, cv=folds, scoring='neg_mean_squared_error').fit(Xs, df[target])
    lasso = LassoCV(alphas=ALPHAS, cv=folds, max_iter=10000).fit(Xs, df[target])
    table = pd.DataFrame({'OLS': ols.coef_,
                          'Ridge': ridge.coef_ / scaler.scale_,
                          'Lasso': lasso.coef_ / scaler.scale_}, index=features).round(4)
    print(f"\nCoefficients on all 41 cities (original units). Ridge alpha = {ridge.alpha_:.3f}, Lasso alpha = {lasso.alpha_:.3f}")
    print(table)
    print("Predictors set to zero by Lasso:", [f for f, c in zip(features, lasso.coef_) if c == 0] or "none")


INTERPRETATION = """
What the penalty does. Ordinary regression picks the coefficients that fit the training data best. Ridge and Lasso add
a penalty that pulls the coefficients towards zero, controlled by alpha: Ridge shrinks all coefficients smoothly and
shares weight between correlated predictors; Lasso can shrink a coefficient exactly to zero, which removes the
predictor. Either can help when there are many predictors, correlated predictors (manufacturing and population have
r = 0.96) or little data (only 41 independent cities).

Choosing alpha. Alpha was chosen by 5-fold cross-validation inside the 40 training cities of every leave-one-city-out
fit, with the folds split by city, so the held-out city never influences the choice and quarters of the same city never
sit on both sides of a split. Lasso's chosen alpha is stable (median 0.2 on standardised predictors). Ridge's chosen
alpha jumps around between folds (median 11, from about 0 to 46), because the cross-validation curve is almost flat:
many alpha values are about equally good.

Coefficients (all 41 cities, original units). Lasso sets the Q4 indicator to exactly zero and roughly halves Q2 and Q3,
which fits the finding that season adds little. Ridge shrinks manufacturing (0.070 to 0.048) and population (-0.046
to -0.025) together, as expected for a correlated pair; the other coefficients move only a little. The signs and
ordering of the important effects (temperature negative, manufacturing positive, population negative) do not change.

Performance for a new city (leave-one-city-out). Ordinary regression gives RMSE 16.03 / R-squared 0.553 at k = 0.
Lasso (16.10 / 0.549) and Ridge (16.39 / 0.533) do not beat it; at k = 1 and 2 Lasso is marginally better
(2.64 and 2.04 against 2.75 and 2.09) and Ridge is worse. These differences are tiny compared with the uncertainty
that comes from having only 41 cities (a 95% interval for the k = 0 RMSE is roughly 12 to 20), so the honest
conclusion is that shrinkage does not improve prediction here. With nine predictors and 41 cities, the limit is the
number of cities, not the number of predictors.

Note: windspeed is kept, as in the other models, although its Memphis and Charleston values are impossible
(about 92 and 65 mph).
"""


def main():
    # 1. Load data and prepare features
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'quarterly_SO2_data.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(script_dir, 'quarterly_SO2_data', 'quarterly_SO2_data.csv')
    if not os.path.exists(data_path):
        data_path = 'quarterly_SO2_data.csv'
    df = pd.read_csv(data_path)

    # Create Quarter dummies (Q1 is the reference)
    df['Q2'] = (df['quarter'] == 2).astype(int)
    df['Q3'] = (df['quarter'] == 3).astype(int)
    df['Q4'] = (df['quarter'] == 4).astype(int)

    features = [
        'temperature',
        'windspeed',
        'precipitation',
        'precipitationdays',
        'manufacturing_enterprises',
        'population_thousands',
        'Q2',
        'Q3',
        'Q4',
    ]
    target = 'SO2'

    # 2. Run evaluation
    evaluate_penalized_models(df, features, target)
    compare_coefficients(df, features, target)


if __name__ == '__main__':
    main()
