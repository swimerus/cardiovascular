import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import optuna
import mlflow
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import from_pandas

mlflow.set_tracking_uri("http://localhost:8080")

df = pd.read_csv("data/preprocessed/preprocessed_data.csv")

X = df.iloc[:, :-1]
y = df['target']

X_dataset = from_pandas(X, source="Cardiovascular_Disease_Dataset_X")
y_dataset = from_pandas(pd.DataFrame(y), source="Cardiovascular_Disease_Dataset_y")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)



def objective(trial):
    with mlflow.start_run(nested=True):
        trial_params = {}
        trial_params["n_neighbors"] = trial.suggest_int("n_neighbors", 2, 10, log=True)
        trial_params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        trial_params["algorithm"] = trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree", "brute"])

        clf = KNeighborsClassifier(**trial_params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accurary", accuracy)
        mlflow.log_params(trial_params)

        return accuracy

mlflow.set_experiment("Cardiovascular with KNN")

with mlflow.start_run():
    mlflow.log_input(X_dataset, "X")
    mlflow.log_input(y_dataset, "y")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    best_trial = study.best_trial
    print(f"Best trial (accuracy): {best_trial.value}")
    print("Best trial (params):")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    mlflow.log_params(best_trial.params)

    model = KNeighborsClassifier(**best_trial.params)
    model.fit(X_train, y_train)

    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="knn-model",
        signature=signature,
        input_example=X_train[:5],
        registered_model_name="cardiovascular_knn"
    )

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
