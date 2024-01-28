# POC KNN Classifier using optuna, mlflow, dvc and bentoml
dataset: https://www.kaggle.com/datasets/jocelyndumlao/cardiovascular-disease-dataset/data

## Building instructions
1. Start mlflow server `mlflow server --host 127.0.0.1 --port 8080`
2. Initilize git and dvc repos
3. Run python scirpts:
   1. `01_preprocessing.py`
   2. `02_model_train.py`
   3. `03_load_model_and_prepare_bentoml_model.py`
4. Run bento server to if everything is OK 
   
   `bentoml serve service:service`
   Example input: `[53,1,2,171,0,0,1,147,0,5.3,3,3]`
5. Build a Bento `bentoml build`
6. Build docker with Bento `bentoml containerize cardiovascular_knn:latest`
7. Run docker image `docker run -p 3000:3000 cardiovascular_knn:kznn2uf5zwr5hd74`

## Limitations
* Only single record is accepted as an input

## Screenshots
### Experiments in mlflow
![mlflow](/images/mlflow_1.png)

![mlflow](/images/mlflow_2.png)

### Project structure
![project structure](/images/project_structure.png)

### BentoML
![bentoML](/images/bentoML.png)



