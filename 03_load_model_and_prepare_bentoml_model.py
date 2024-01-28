import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import bentoml

mlflow.set_tracking_uri("http://localhost:8080")


model = mlflow.sklearn.load_model(model_uri="models:/cardiovascular_knn/2")

df=pd.read_csv("data/preprocessed/preprocessed_data.csv")
X = df.iloc[:, :-1]
y = df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_val)
print(accuracy_score(y_val,y_pred))



bentoml.picklable_model.save_model(
    "Cardiovascular_KNN",
    model,
    signatures={"predict": {"batchable": False}}
)