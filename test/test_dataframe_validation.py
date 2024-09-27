import numpy as np
import pandas as pd

from pandas_schema import Schema, Column
from pandas_schema.validation_df import IsDtypeValidation, DuplicatedRowsValidation


df_schema = Schema(columns=[
    Column('Required', validations=[IsDtypeValidation(dtype=np.dtype(np.int64))]),
    Column('Not Required', allow_empty=True),
], validations=[DuplicatedRowsValidation()])


def test_validate_single_column_in_dataframe_with_no_error():
    df = pd.DataFrame({'Not Required': ['Tom', 'nick', '', 'jack'],
                       'Required': [20, 21, 19, 18]})
    errors = df_schema.validate(df)

    assert not errors


def test_validate_single_column_in_dataframe_with_error():
    df = pd.DataFrame({'Not Required': ['Tom', 'nick', '', 'jack'],
                       'Required': [20, 21, None, 18]})
    errors = df_schema.validate(df)

    assert len(errors) == 4
    assert errors[0].column == 'Required'
    assert errors[0].message == 'contains dtype which is not a subclass of the required type "int64"'


def test_validate_duplicates_in_dataframe():
    df = pd.DataFrame({'Not Required': ['Tom', 'nick', 'Tom', 'jack'],
                       'Required': [20, 21, 20, 18]})
    errors = df_schema.validate(df)

    assert len(errors) == 1
    assert errors[0].column == 'Not Required, Required'
    assert errors[0].row == 2
    assert errors[0].message == 'Duplicated'
