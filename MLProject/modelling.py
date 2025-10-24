import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    args = parser.parse_args()

    # Load data
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)

    X_train = train.drop(columns=["44"])
    y_train = train["44"]
    X_test = test.drop(columns=["44"])
    y_test = test["44"]

    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"Accuracy: {acc:.4f}")
        mlflow.log_metric("accuracy", acc)