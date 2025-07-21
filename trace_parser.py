import pandas as pd
import ast
import json
import csv
from typing import Dict, List
import yaml
import numpy as np
import os
import gzip

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

    
