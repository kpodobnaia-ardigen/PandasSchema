import abc
import math
import datetime
import pandas as pd
import numpy as np
import typing
import operator

from . import column
from .validation_warning import ValidationWarning
from .errors import PanSchArgumentError
from pandas.api.types import is_categorical_dtype, is_numeric_dtype


class _BaseValidation:
    """
    The validation base class that defines any object that can create a list of errors from a Series
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_errors(self, df: pd.DataFrame) -> typing.Iterable[ValidationWarning]:
        """
        Return a list of errors in the given series
        :param df:
        :return:
        """


class _DataFrameValidation(_BaseValidation):
    """
    Implements the _BaseValidation interface by returning a Boolean series for each element that
    either passes or fails the validation
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, columns=None, **kwargs):
        self._custom_message = kwargs.get('error_message')
        self.columns = columns

    @property
    def message(self):
        return self._custom_message or self.default_message

    @property
    @abc.abstractmethod
    def default_message(self) -> str:
        """
        Create a message to be displayed whenever this validation fails
        This should be a generic message for the validation type, but can be overwritten if the user provides a
        message kwarg
        """

    @abc.abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns a Boolean series, where each value of False is an element in the Series
        that has failed the validation
        :param df:
        :return:
        """

    def __invert__(self):
        """
        Returns a negated version of this validation
        """
        return _InverseValidation(self)

    def __or__(self, other: '_DataFrameValidation'):
        """
        Returns a validation which is true if either this or the other validation is true
        """
        return _CombinedValidation(self, other, operator.or_)

    def __and__(self, other: '_DataFrameValidation'):
        """
        Returns a validation which is true if either this or the other validation is true
        """
        return _CombinedValidation(self, other, operator.and_)

    def get_errors(self, df: pd.DataFrame):
        if self.columns:
            df = df[self.columns]

        validation_results_per_row = ~self.validate(df)

        invalid_rows = df[validation_results_per_row]

        if invalid_rows.empty:
            return []

        errors = []
        for index, row in invalid_rows.iterrows():
            errors.append(ValidationWarning(
                message=self.message,
                value=row,
                row=index,
                column=', '.join(list(df.keys()))
            ))

        return errors


class _InverseValidation(_DataFrameValidation):
    """
    Negates an ElementValidation
    """

    def __init__(self, validation: _DataFrameValidation):
        self.negated = validation
        super().__init__()

    def validate(self, df: pd.DataFrame):
        return ~ self.negated.validate(df)

    @property
    def default_message(self):
        return self.negated.message + ' <negated>'


class _CombinedValidation(_DataFrameValidation):
    """
    Validates if one and/or the other validation is true for an element
    """

    def __init__(self, validation_a: _DataFrameValidation, validation_b: _DataFrameValidation, operator):
        self.operator = operator
        self.v_a = validation_a
        self.v_b = validation_b
        super().__init__()

    def validate(self, df: pd.DataFrame):
        return self.operator(self.v_a.validate(df), self.v_b.validate(df))

    @property
    def default_message(self):
        return '({}) {} ({})'.format(self.v_a.message, self.operator, self.v_b.message)


class IsDtypeValidation(_DataFrameValidation):
    """
    Checks that a series has a certain numpy dtype
    """

    def __init__(self, dtype: np.dtype, **kwargs):
        """
        :param dtype: The numpy dtype to check the column against
        """
        self.dtype = dtype
        super().__init__(**kwargs)

    def validate(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(np.issubdtype, axis=1, arg2=np.dtype('int64'))

    @property
    def default_message(self):
        return f'contains dtype which is not a subclass of the required type "{self.dtype}"'


class DuplicatedRowsValidation(_DataFrameValidation):
    """
    Checks duplicated rows:
        - by default all columns are taken into account
        - a subset of columns can be taken into account if specified
    """

    @property
    def default_message(self):
        return 'Duplicated'

    def validate(self, df: pd.DataFrame) -> pd.Series:
        return ~df.duplicated(subset=self.columns or df.columns)
