import pandas as pd
from typing import List, Any, Dict, Optional


class DataFrameGenerator:
    """A utility class for building pandas DataFrames from column data."""
    
    def __init__(self):
        """Initialize an empty DataFrameGenerator."""
        self.data: Dict[str, List[Any]] = {}

    def add_data(self, column_name: str, values: List[Any]) -> None:
        """Add data to a specific column.
        
        Args:
            column_name: Name of the column to add data to
            values: List of values to add to the column
        """
        if not isinstance(column_name, str):
            raise ValueError("column_name must be a string")
        if not isinstance(values, list):
            raise ValueError("values must be a list")
        
        if column_name not in self.data:
            self.data[column_name] = []
        self.data[column_name].extend(values)

    def add_single_value(self, column_name: str, value: Any) -> None:
        """Add a single value to a specific column.
        
        Args:
            column_name: Name of the column to add data to
            value: Single value to add to the column
        """
        self.add_data(column_name, [value])

    def get_column_lengths(self) -> Dict[str, int]:
        """Get the length of each column.
        
        Returns:
            Dictionary mapping column names to their lengths
        """
        return {col: len(values) for col, values in self.data.items()}

    def is_balanced(self) -> bool:
        """Check if all columns have the same length.
        
        Returns:
            True if all columns have the same length, False otherwise
        """
        if not self.data:
            return True
        lengths = set(len(col) for col in self.data.values())
        return len(lengths) == 1

    def to_dataframe(self, auto_balance: bool = True) -> pd.DataFrame:
        """Convert the stored data to a pandas DataFrame.
        
        Args:
            auto_balance: If True, automatically trim columns to the minimum length.
                        If False, raise an error if columns have different lengths.
        
        Returns:
            pandas DataFrame with the stored data
            
        Raises:
            ValueError: If auto_balance is False and columns have different lengths
        """
        if not self.data:
            return pd.DataFrame()
        
        if not auto_balance and not self.is_balanced():
            lengths = self.get_column_lengths()
            raise ValueError(f"Columns have different lengths: {lengths}")
        
        # Find the minimum length among all columns
        min_len = min(len(col) for col in self.data.values())
        
        # Trim each column to the minimum length
        trimmed_data = {k: v[:min_len] for k, v in self.data.items()}
        
        return pd.DataFrame(trimmed_data)

    def clear(self) -> None:
        """Clear all stored data."""
        self.data.clear()

    def get_column_names(self) -> List[str]:
        """Get the names of all columns.
        
        Returns:
            List of column names
        """
        return list(self.data.keys())

    def has_column(self, column_name: str) -> bool:
        """Check if a column exists.
        
        Args:
            column_name: Name of the column to check
            
        Returns:
            True if the column exists, False otherwise
        """
        return column_name in self.data

    def merge(self, other_dataframe_generator: "DataFrameGenerator"):
        """Merge the stored data with another DataFrameGenerator.
        
        Args:
            other_dataframe_generator: Another DataFrameGenerator to merge with
            
        Returns:
            Merged DataFrameGenerator
        """
        if not isinstance(other_dataframe_generator, DataFrameGenerator):
            raise ValueError("other_dataframe_generator must be a DataFrameGenerator")
        # Check if this DataFrameGenerator is empty
        if not self.data:
            self.data = other_dataframe_generator.data
            return
        # Check if the other DataFrameGenerator has the same column names
        if not set(self.get_column_names()) == set(other_dataframe_generator.get_column_names()):
            print("The two DataFrameGenerators have different column names")
            return
            # raise ValueError("The two DataFrameGenerators have different column names")
        # Merge the data
        for column_name in other_dataframe_generator.get_column_names():
            self.add_data(column_name, other_dataframe_generator.data[column_name])
