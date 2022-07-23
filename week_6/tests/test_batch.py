#!/usr/bin/env python
# coding: utf-8

from datetime import datetime

import pandas as pd
from deepdiff import DeepDiff

import batch


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]
    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    raw_df = pd.DataFrame(data, columns=columns)

    prepared_df = batch.prepare_data(raw_df, ['PUlocationID', 'DOlocationID'])

    expected_data = [
        ('-1', '-1', dt(1, 2), dt(1, 10), 8.0),
        ('1', '1', dt(1, 2), dt(1, 10), 8.0)
    ]

    expected_columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime', 'duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)

    diff = DeepDiff(expected_df.to_dict('records'), prepared_df.to_dict('records'), significant_digits=1)

    assert 'values_changed' not in diff
    assert 'type_changes' not in diff
