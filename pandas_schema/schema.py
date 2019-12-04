import pandas as pd
import typing

from . import validation_df
from .errors import PanSchInvalidSchemaError, PanSchArgumentError
from .validation_warning import ValidationWarning
from .column import Column


class Schema:
    """
    A schema that defines the columns required in the target DataFrame

    :param columns: A list of column objects
    :param validations: A list of Schema-level validations
    :param ordered: True if the Schema should associate its Columns with DataFrame columns
                    by position only, ignoring the header names. False if the columns should be
                    associated by column header names only. Defaults to False
    """

    def __init__(self, columns: typing.Iterable[Column],
                 validations: typing.Iterable['validation_df._BaseValidation'] = [],
                 ordered: bool = False):
        if not columns:
            raise PanSchInvalidSchemaError('An instance of the schema class must have a columns list')

        if not isinstance(columns, typing.List):
            raise PanSchInvalidSchemaError('The columns field must be a list of Column objects')

        if not isinstance(ordered, bool):
            raise PanSchInvalidSchemaError('The ordered field must be a boolean')

        self.columns = list(columns)
        self.ordered = ordered
        self.validations = list(validations)

    def validate(self, df: pd.DataFrame, columns: typing.List[str] = None) -> typing.List[ValidationWarning]:
        """
        Runs a full validation of the target DataFrame using the internal columns list

        :param df: A pandas DataFrame to validate
        :param columns: A list of columns indicating a subset of the schema that we want to validate
        :return: A list of ValidationWarning objects that list the ways in which the DataFrame was invalid
        """
        errors = []

        # TODO: headers validations

        if columns:
            if not set(columns).issubset(self.get_column_names()):
                raise PanSchArgumentError(
                    'Columns {} passed in are not part of the schema'.format(
                        set(columns).difference(self.columns))
                )
            validating_columns = [column for column in self.columns if column.name in columns]
        else:
            validating_columns = self.columns
            # self._validate_number_of_columns(df)

        for schema_column in validating_columns:
            errors += schema_column.validate(df)

        for schema_validation in self.validations:
            errors += schema_validation.get_errors(df)

        return sorted(errors, key=lambda e: e.row)

    def _validate_number_of_columns(self, df):
        schema_cols = len(self.columns)
        df_cols = len(df.columns)

        if df_cols != schema_cols:
            return [ValidationWarning('Invalid number of columns. The schema specifies {}, '
                                      'but the data frame has {}'.format(schema_cols, df_cols))]

    def _validate_columns_in_schema(self):
        pass

    def get_column_names(self):
        """
        Returns the column names contained in the schema
        """
        return [column.name for column in self.columns]
