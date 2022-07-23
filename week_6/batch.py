#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import sys
from typing import List, NoReturn

import pandas as pd
import requests


def run() -> NoReturn:
    try:
        year = int(sys.argv[1])
    except IndexError:
        year = 2021
    try:
        month = int(sys.argv[2])
    except IndexError:
        month = 2

    main(year, month)


def main(year: int, month: int) -> NoReturn:
    categorical = ['PUlocationID', 'DOlocationID']

    input_file_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_{year:04d}-{month:02d}.parquet"
    output_file_name = f'predictions_{year:04d}-{month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)


    df = prepare_data(read_data(input_file_url), categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    output_file = os.path.join(os.getcwd(), output_file_name)
    if os.path.exists(output_file):
        os.remove(output_file)

    df_result.to_parquet(output_file, engine='pyarrow', index=False)


def prepare_data(raw_df: pd.DataFrame, categorical: List[str]) -> pd.DataFrame:
    raw_df['duration'] = raw_df.dropOff_datetime - raw_df.pickup_datetime
    raw_df['duration'] = raw_df.duration.dt.total_seconds() / 60

    raw_df = raw_df[(raw_df.duration >= 1) & (raw_df.duration <= 60)].copy()

    raw_df[categorical] = raw_df[categorical].fillna(-1).astype('int').astype('str')
    
    return raw_df


def read_data(input_file_url: str) -> pd.DataFrame:
    file_path = os.path.join(os.getcwd(), input_file_url.split('/')[-1])
    if os.path.exists(file_path):
        os.remove(file_path)

    response = requests.get(input_file_url)
    with open(file_path, 'wb') as f:
        f.write(response.content)

    df = pd.read_parquet(file_path)

    return df
    

if __name__ == '__main__':
    run()