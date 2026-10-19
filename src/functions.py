from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def load_data():
    """Load the cell growth dataset from the datasets directory."""

    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "datasets" / "dataset_cell_growth_4factor.csv"

    data = pd.read_csv(file_path)

    return data

def split_features_target(data):
    """Separate input features from the target growth rate."""

    feature_columns = [
        "Temperature [C]",
        "pH",
        "Glucose [mM]",
        "Lactate [mM]"
    ]

    target_column = "Specific Growth Rate [d^-1]"

    X = data[feature_columns]
    y = data[target_column]

    return X, y

from sklearn.model_selection import train_test_split


def prepare_data(X, y):
    """Split data into training and testing sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test

def train_polynomial_model(X_train, y_train):
    """Train a second-degree polynomial regression model."""

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    return model

def train_knn_model(X_train, y_train):
    """Train a K-Nearest Neighbors regression model."""

    model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(
            n_neighbors=5,
            weights="distance"
        )
    )

    model.fit(X_train, y_train)

    return model

def optimize_knn_model(X_train, y_train):
    """Optimize KNN hyperparameters using cross-validation."""

    pipeline = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor()
    )

    parameter_grid = {
        "kneighborsregressor__n_neighbors": [3, 5, 7, 9, 11, 15],
        "kneighborsregressor__weights": ["uniform", "distance"]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search

def optimize_polynomial_model(X_train, y_train):
    """Optimize polynomial regression using cross-validation."""

    pipeline = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(include_bias=False),
        LinearRegression()
    )

    parameter_grid = {
        "polynomialfeatures__degree": [1, 2, 3, 4]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search

def plot_actual_vs_predicted(y_test, y_pred):
    """Plot actual versus predicted specific growth rates."""

    project_root = Path(__file__).resolve().parent.parent
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.7,
        edgecolors="black"
    )

    lower = min(y_test.min(), y_pred.min())
    upper = max(y_test.max(), y_pred.max())

    plt.plot(
        [lower, upper],
        [lower, upper],
        "r--",
        label="Perfect prediction"
    )

    plt.xlabel("Actual Growth Rate (d$^{-1}$)")
    plt.ylabel("Predicted Growth Rate (d$^{-1}$)")
    plt.title("Polynomial Regression: Actual vs Predicted")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        figures_dir / "actual_vs_predicted.png",
        dpi=300
    )

    plt.close()

def plot_residuals(y_test, y_pred):
    """Plot prediction residuals against predicted growth rates."""

    project_root = Path(__file__).resolve().parent.parent
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    residuals = y_test - y_pred

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.7,
        edgecolors="black"
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
        label="Zero residual"
    )

    plt.xlabel("Predicted Growth Rate (d$^{-1}$)")
    plt.ylabel("Residual (d$^{-1}$)")
    plt.title("Polynomial Regression: Residual Analysis")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        figures_dir / "residual_analysis.png",
        dpi=300
    )

    plt.close()

def save_model_comparison(knn_search, poly_search, rf_search, X_test, y_test):
    """Export optimized model performance to a CSV file."""

    project_root = Path(__file__).resolve().parent.parent
    tables_dir = project_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "KNN Regression": knn_search,
        "Polynomial Regression": poly_search,
        "Random Forest Regression": rf_search
    }

    results = []

    for name, search in models.items():
        predictions = search.best_estimator_.predict(X_test)

        results.append({
            "Model": name,
            "CV RMSE (d^-1)": -search.best_score_,
            "Test MAE (d^-1)": mean_absolute_error(y_test, predictions),
            "Test RMSE (d^-1)": np.sqrt(
                mean_squared_error(y_test, predictions)
            ),
            "Test R2": r2_score(y_test, predictions),
            "Best Parameters": str(search.best_params_)
        })

    comparison = pd.DataFrame(results)

    comparison.to_csv(
        tables_dir / "model_comparison.csv",
        index=False
    )

    return comparison

def plot_feature_relationships(data):
    """Visualize relationships between process variables and growth rate."""

    project_root = Path(__file__).resolve().parent.parent
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    features = [
        "Temperature [C]",
        "pH",
        "Glucose [mM]",
        "Lactate [mM]"
    ]

    target = "Specific Growth Rate [d^-1]"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, feature in zip(axes.flatten(), features):
        ax.scatter(
            data[feature],
            data[target],
            alpha=0.6,
            edgecolors="black"
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("Specific Growth Rate (d$^{-1}$)")
        ax.set_title(f"Growth Rate vs {feature}")
        ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        figures_dir / "feature_relationships.png",
        dpi=300
    )

    plt.close()

def train_random_forest(X_train, y_train):
    """Train a Random Forest regression model."""

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

def optimize_random_forest(X_train, y_train):
    """Optimize Random Forest using cross-validation."""

    model = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [2, 4, 6]
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search

