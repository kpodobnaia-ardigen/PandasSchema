import abc
import pandas as pd
import numpy as np
import typing
import operator

from .validation_warning import ValidationWarning
from .errors import PanSchArgumentError


class _BaseValidation:
    """
    The validation base class that defines any object that can create a list of errors from a Series
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_errors(self, df: pd.DataFrame, allow_empty) -> typing.Iterable[ValidationWarning]:
        """
        Return a list of errors in the given series
        :param df:
        :param allow_empty:
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

    def get_errors(self, df: pd.DataFrame, allow_empty: bool = False):
        if self.columns:
            df = df[self.columns]

        validation_results_per_row = ~self.validate(df)

        if allow_empty:
            empty_validation_results = EmptyValuesValidation().validate(df)
            validation_results_per_row = empty_validation_results & validation_results_per_row

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


class CanCallValidation(_DataFrameValidation):
    """
    Validates if a given function can be called on each element in a column without raising an exception
    """

    def __init__(self, func: typing.Callable, **kwargs):
        """
        :param func: A python function that will be called with the value of each cell in the DataFrame. If this
            function throws an error, this cell is considered to have failed the validation. Otherwise it has passed.
        """
        if callable(func):
            self.callable = func
        else:
            raise PanSchArgumentError('The object "{}" passed to CanCallValidation is not callable!'.format(type))
        super().__init__(**kwargs)

    @property
    def default_message(self):
        return 'raised an exception when the callable {} was called on it'.format(self.callable)

    def can_call(self, func_arg):
        try:
            self.callable(func_arg)
            return True
        except:
            return False

    def validate(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(self.can_call, axis=1)


class CanConvertValidation(CanCallValidation):
    """
    Checks if each element in a column can be converted to a Python object type
    """

    """
    Internally this uses the same logic as CanCallValidation since all types are callable in python.
    However this class overrides the error messages to make them more directed towards types
    """

    def __init__(self, _type: type, **kwargs):
        """
        :param _type: Any python type. Its constructor will be called with the value of the individual cell as its
            only argument. If it throws an exception, the value is considered to fail the validation, otherwise it has passed
        """
        if isinstance(_type, type):
            super().__init__(_type, **kwargs)
        else:
            raise PanSchArgumentError('{} is not a valid type'.format(_type))

    @property
    def default_message(self):
        return 'cannot be converted to type {}'.format(self.callable)


class MatchesPatternValidation(_DataFrameValidation):
    """
    Validates that a string or regular expression can match somewhere in each element in this column
    """

    def __init__(self, pattern, options=None, **kwargs):
        """
        :param kwargs: Arguments to pass to Series.str.contains
            (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.contains.html)
            pat is the only required argument
        """
        self.pattern = pattern
        self.options = options or {}

        super().__init__(**kwargs)

    @property
    def default_message(self):
        return 'does not match the pattern "{}"'.format(self.pattern)

    def validate(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(lambda series: series.astype(str).str.contains(self.pattern, **self.options), axis=1)


class LeadingWhitespaceValidation(MatchesPatternValidation):
    """
    Checks that there is no leading whitespace in this column
    """
    def __init__(self, **kwargs):
        super().__init__(pattern=r'^\s+', **kwargs)

    @property
    def default_message(self):
        return 'contains leading whitespace'


class TrailingWhitespaceValidation(MatchesPatternValidation):
    """
    Checks that there is no trailing whitespace in this column
    """

    def __init__(self, **kwargs):
        super().__init__(pattern=r'\s+$', **kwargs)

    @property
    def default_message(self):
        return 'contains trailing whitespace'


class EmptyValuesValidation(_DataFrameValidation):
    """
    Returns False when there is an empty value in a row.
    """
    def validate(self, df: pd.DataFrame) -> pd.Series:
        df = df.replace(r'^\s*$', np.nan, regex=True)
        return ~df.isnull().all(axis=1)
