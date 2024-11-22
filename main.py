import click
import mlflow
import mlflow.sklearn
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# Metrics we will plot to evaluate
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

@click.command()
@click.option("--learning_rate", default=0.1)
@click.option("--n_estimators", default=100)
@click.option("--max_depth", default=3)
def main(learning_rate, n_estimators, max_depth):
    print('Starting')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0.2, seed=113
    )

    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Model
    # Create the model using the parameters
    # Create the model using the parameters
    model = GradientBoostingRegressor(loss='squared_error',
                                      learning_rate=learning_rate,
                                      n_estimators=n_estimators,
                                      max_depth=max_depth)

    # Fit the model to the data
    model.fit(x_train, y_train)

    # Predict the test data
    y_pred = model.predict(x_test)

    # Check the metrics (real vs predicted)
    rmse_test, mae_test, r2_test = eval_metrics(y_test, y_pred)

    # Log the metrics to MLFlow
    mlflow.log_metric("rmse", rmse_test)
    mlflow.log_metric("mae", mae_test)
    mlflow.log_metric("r2", r2_test)
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
