import pandas as pd
import ast
import json
import csv
from typing import Dict, List, Any, Optional
import yaml
import numpy as np
import os
import gzip
import jax
import jax.numpy as jnp
from bidict import bidict
import statistics


JAX_DTYPE_TO_STRING = bidict[jnp.dtype, str]({
    jnp.float32: "float32",
    jnp.float64: "float64",
    jnp.int8: "int8",
    jnp.int16: "int16",
    jnp.int32: "int32",
    jnp.int64: "int64",
})


class TraceParser:
    def __init__(self, trace_dir: str):
        self.trace_dir = trace_dir

    def set_trace_dir(self, new_dir: str):
        """
        Set a new trace directory for the parser.
        """
        self.trace_dir = new_dir

    def find_trace_file(self):
        """
        Recursively search for the first .trace.json.gz file in the trace_dir.
        Returns the full path to the file, or None if not found.
        """
        for root, _, files in os.walk(self.trace_dir):
            for file in files:
                if file.endswith('.trace.json.gz'):
                    return os.path.join(root, file)
        return None

    def read_trace_json(self):
        """
        Finds, unzips, and reads the JSON content from the trace file.
        Returns the loaded JSON object, or None if not found or error.
        """
        trace_file = self.find_trace_file()
        if trace_file is None:
            print("No trace file found.")
            return None
        try:
            with gzip.open(trace_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error reading trace file: {e}")
            return None

    def parse_trace_csv(self):
        """
        Parses the trace CSV file and returns a list of trace events.
        """
        csv_file = os.path.join(self.trace_dir, 'trace_events.csv')
        
        # Read the trace JSON data
        trace_data = self.read_trace_json()
        if trace_data is None:
            print("Failed to read trace data")
            return None
            
        # Extract trace events
        trace_events = trace_data.get('traceEvents', [])
        if not trace_events:
            print("No trace events found in the data")
            return None

        headers = ['pid', 'tid', 'ts', 'dur', 'ph', 'name', 'args']
        # Write to CSV directly
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for event in trace_events:
                # Convert args dictionary to string if it exists
                if 'args' in event:
                    event['args'] = json.dumps(event['args'])
                else:
                    event['args'] = ''
                
                # Write the event
                writer.writerow(event)
        print(f"Trace events written to: {csv_file}")
        # return trace_events

    

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

    
    def get_header(self) -> List[str]:
        """Get the header of the DataFrameGenerator.
        
        Returns:
            List of column names
        """
        return list(self.data.keys())
    
    def get_row_dict(self, index: int) -> List[Any]:
        """Get a row of the DataFrameGenerator.
        
        Returns:
            Dictionary of column names and values
        """
        return {column_name: self.data[column_name][index] for column_name in self.get_column_names()}



def calculate_statistics(data: List[Any]) -> Dict[str, Any]:
    """Calculate the statistics of the data.
    
    Args:
        data: List of data
        
    Returns:
        Dictionary containing the statistics
    """
    return {
        "mean": statistics.mean(data),
        "std": statistics.stdev(data),
        "min": min(data),
        "max": max(data),
        "median": statistics.median(data),
    }


def list_add(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Sum two lists element-wise.
    
    Args:
        list1: First list to sum
        list2: Second list to sum
        
    Returns:
        List of the sum of the two lists
    """
    assert len(list1) == len(list2), "The two lists must have the same length"
    return [e1 + e2 for e1, e2 in zip(list1, list2)]