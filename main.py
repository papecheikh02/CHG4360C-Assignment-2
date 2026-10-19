from src.functions import (
    load_data,
    split_features_target,
    prepare_data,
    train_polynomial_model,
    train_knn_model,
    optimize_knn_model,
    optimize_polynomial_model,
    plot_actual_vs_predicted,
    plot_residuals,
    save_model_comparison,
    plot_feature_relationships,
    train_random_forest,
    optimize_random_forest
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def main():
    # Load the cell growth dataset
    data = load_data()

    # Separate input features and target variable
    X, y = split_features_target(data)

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Train the polynomial regression model
    model = train_polynomial_model(X_train, y_train)

    # Predict growth rates using test data
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Display model performance
    print("\nPolynomial Regression Performance:")
    print(f"MAE: {mae:.4f} d^-1")
    print(f"RMSE: {rmse:.4f} d^-1")
    print(f"R² Score: {r2:.4f}")

    # Train the KNN regression model
    knn_model = train_knn_model(X_train, y_train)

    # Predict growth rates using KNN
    knn_pred = knn_model.predict(X_test)

    # Calculate KNN performance metrics
    knn_mae = mean_absolute_error(y_test, knn_pred)
    knn_rmse = np.sqrt(mean_squared_error(y_test, knn_pred))
    knn_r2 = r2_score(y_test, knn_pred)

    # Display KNN performance
    print("\nKNN Regression Performance:")
    print(f"MAE: {knn_mae:.4f} d^-1")
    print(f"RMSE: {knn_rmse:.4f} d^-1")
    print(f"R² Score: {knn_r2:.4f}")

    # Optimize KNN using cross-validation
    print("\nOptimizing KNN model...")
    knn_search = optimize_knn_model(X_train, y_train)

    print("Best KNN parameters:", knn_search.best_params_)
    print(f"Best KNN CV RMSE: {-knn_search.best_score_:.4f} d^-1")

    # Optimize polynomial regression using cross-validation
    print("\nOptimizing Polynomial Regression...")
    poly_search = optimize_polynomial_model(X_train, y_train)

    print("Best Polynomial parameters:", poly_search.best_params_)
    print(f"Best Polynomial CV RMSE: {-poly_search.best_score_:.4f} d^-1")



    # Generate actual vs predicted growth rate figure
    final_model = poly_search.best_estimator_
    final_predictions = final_model.predict(X_test)
    plot_actual_vs_predicted(y_test, final_predictions)
    print("\nFigure saved: figures/actual_vs_predicted.png")

    # Generate residual analysis figure
    plot_residuals(y_test, final_predictions)
    print("Figure saved: figures/residual_analysis.png")


    # Visualize relationships between culture conditions and growth
    plot_feature_relationships(data)
    print("\nFigure saved: figures/feature_relationships.png")


    # Train Random Forest model
    print("\nTraining Random Forest Regression...")

    rf_model = train_random_forest(X_train, y_train)

    # Predict growth rates
    rf_predictions = rf_model.predict(X_test)

    # Evaluate model performance
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    rf_r2 = r2_score(y_test, rf_predictions)
    print("\nRandom Forest Regression Performance:")
    print(f"MAE: {rf_mae:.4f} d^-1")
    print(f"RMSE: {rf_rmse:.4f} d^-1")
    print(f"R² Score: {rf_r2:.4f}")



    # Optimize Random Forest using cross-validation
    print("\nOptimizing Random Forest Regression...")

    rf_search = optimize_random_forest(X_train, y_train)

    print("Best Random Forest parameters:", rf_search.best_params_)
    print(f"Best Random Forest CV RMSE: {-rf_search.best_score_:.4f} d^-1")

    # Evaluate optimized model on the test dataset
    rf_optimized = rf_search.best_estimator_
    rf_optimized_predictions = rf_optimized.predict(X_test)

    rf_opt_mae = mean_absolute_error(y_test, rf_optimized_predictions)
    rf_opt_rmse = np.sqrt(
        mean_squared_error(y_test, rf_optimized_predictions)
    )
    rf_opt_r2 = r2_score(y_test, rf_optimized_predictions)
    print("\nOptimized Random Forest Performance:")
    print(f"MAE: {rf_opt_mae:.4f} d^-1")
    print(f"RMSE: {rf_opt_rmse:.4f} d^-1")
    print(f"R² Score: {rf_opt_r2:.4f}")

    # Export comparison of all three optimized models
    comparison_table = save_model_comparison(
        knn_search,
        poly_search,
        rf_search,
        X_test,
        y_test
    )
    print("\nFinal Model Comparison Table:")
    print(comparison_table.to_string(index=False))
    print("\nTable saved: tables/model_comparison.csv")


    # Display dataset dimensions
    print("Total observations:", len(data))
    print("Training observations:", len(X_train))
    print("Testing observations:", len(X_test))

    # Display input features
    print("\nInput features:")
    print(X.columns.tolist())

    # Display target variable
    print("\nTarget variable:")
    print(y.name)


if __name__ == "__main__":
    main()

