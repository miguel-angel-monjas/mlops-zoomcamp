import calendar
import os
from datetime import datetime
from datetime import date as dt

import joblib
import pandas as pd
import requests
from prefect import flow, task, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.task_runners import SequentialTaskRunner
from prefect.orion.schemas.schedules import CronSchedule

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

BASE_TAXI_URL='https://nyc-tlc.s3.amazonaws.com/trip+data'

    
@task
def get_paths(current_date=None):
    logger = get_run_logger()

    if current_date is None:
        current_date = dt.today()
    elif isinstance(current_date, str):
        time_format = '%Y-%m-%d'
        try:
            current_date = datetime.strptime(current_date, time_format)
        except ValueError:
            current_date = dt.today()

    current_month = current_date.month
    training_month = current_month - 2
    validation_month = current_month - 1
    logger.info(f"INFO - Training data from {calendar.month_name[training_month]}")
    logger.info(f"INFO - Validation data from {calendar.month_name[validation_month]}")

    if training_month > 0:
        train_url = f"{BASE_TAXI_URL}/fhv_tripdata_2021-{training_month:02d}.parquet"
    else:
        train_url = f"{BASE_TAXI_URL}/fhv_tripdata_2020-{12+training_month:02d}.parquet"
    if validation_month > 0:
        val_url = f"{BASE_TAXI_URL}/fhv_tripdata_2021-{validation_month:02d}.parquet"
    else:
        val_url = f"{BASE_TAXI_URL}/fhv_tripdata_2020-{12+validation_month:02d}.parquet"

    return train_url, val_url, str(current_date)

@task
def read_data(url: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"INFO - Running read_data from {url}")

    i = url[-9:-8]
    current_folder = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(current_folder, 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    file_path = os.path.join(data_folder, f'fhv_tripdata_2021-0{i}.parquet')
    if not os.path.exists(file_path):
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
    df = pd.read_parquet(file_path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"INFO - The mean duration of training is {round(mean_duration,2)} seconds")
    else:
        logger.info(f"INFO - The mean duration of validation is {round(mean_duration,2)} seconds")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"INFO - The shape of X_train is {X_train.shape}")
    logger.info(f"INFO - The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"INFO - The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"INFO - The MSE of inference is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):
    logger = get_run_logger()
    logger.info(f"INFO - Running flow with date {date}")

    train_dataset_url, validation_dataset_url, current_date = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_dataset_url).result()
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(validation_dataset_url).result()
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    # save the models
    current_folder = os.path.abspath(os.path.dirname(__file__))
    models_folder = os.path.join(current_folder, 'models')
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    model_file_path = os.path.join(models_folder, f'model-{date}.bin')
    dv_file_path = os.path.join(models_folder, f'dv-{date}.bin')

    logger.info(f"INFO - Saving model in {model_file_path}")
    joblib.dump(lr, model_file_path)
    logger.info(f"INFO - Saving model in {dv_file_path}")
    joblib.dump(dv, dv_file_path)

DeploymentSpec(
  flow=main,
  name="model_training",
  schedule=CronSchedule(cron="0 9 15 * *"),
  flow_runner=SubprocessFlowRunner(),
  tags=["prefect","homework","week3"]
)

