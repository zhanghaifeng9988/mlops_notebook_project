
import pandas as pd
import xgboost as xgb
import mlflow
from prefect import flow, task
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
import pickle
from pathlib import Path


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

# ---------- Task 1: 读取数据 ----------
@task
def read_data(year: int, month: int):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

# ---------- Task 2: 特征工程 ----------
@task
def create_features(df: pd.DataFrame, dv: DictVectorizer = None):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']

    train_dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(train_dicts)
    else:
        X = dv.transform(train_dicts)

    y = df['duration'].values
    return X, y, dv

# ---------- Task 3: 训练模型 ----------
@task
def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 20,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:squarederror',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=50,
            evals=[(valid, 'validation')],
            early_stopping_rounds=5
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id

# ---------- Flow: 串联所有 Task ----------
@flow
def training_pipeline(year: int, month: int):
    # 1. 读训练数据
    df_train = read_data(year, month)

    # 2. 立刻做特征工程，然后释放 df_train
    X_train, y_train, dv = create_features(df_train)
    del df_train

    # 3. 读验证数据
    df_val = read_data(year, month + 1) if month < 12 else read_data(year + 1, 1)

    # 4. 用训练集的 dv 处理验证集，然后释放 df_val
    X_val, y_val, _ = create_features(df_val, dv)
    del df_val

    # 5. 训练模型
    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"训练完成，MLflow run_id: {run_id}")

if __name__ == "__main__":
    training_pipeline(2021, 1)