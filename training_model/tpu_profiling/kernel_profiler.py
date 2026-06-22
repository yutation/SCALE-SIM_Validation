import os
from typing import Any, Dict, Optional, Tuple
import jax_kernel_functions as jkf
import jax
import json
from utils import TraceParser, calculate_statistics, list_add, DataFrameGenerator
import warnings
import jax.numpy as jnp
import numpy as np


class KernelProfilerBase:
    def __init__(self,profiler_name:str, kernel_wrapper: jkf.KernelWarpper, trace_dir: str, configuration: Optional[Dict[str, Any]] = {}):
        self.profiler_name: str = profiler_name
        self.kernel_wrapper: jkf.KernelWarpper = kernel_wrapper
        self.trace_dir: str = trace_dir
        self.configuration: Dict[str, Any] = configuration

        self.profile_folder: str = os.path.join(self.trace_dir, self.profiler_name)
        if not os.path.exists(self.profile_folder):
            os.makedirs(self.profile_folder)

    def profile(self):
        pass

    def post_process(self):
        pass

    def get_profiling_dataframe_generator(self):
        pass

class KernelProfilerV0(KernelProfilerBase):
    def __init__(self, profiler_name: str, kernel_wrapper: jkf.KernelWarpper, trace_dir: str, configuration: Optional[Dict[str, Any]] = {}):
        super().__init__(profiler_name, kernel_wrapper, trace_dir, configuration)
        self.random_seed = configuration.get("random_seed", 0)
        self.iterations = configuration.get("iterations", 10)
        self.save_to_file = configuration.get("save_to_file", True)

        self.json_trace_events = None
        self.filtered_events = None
        self.profiling_statistics = None

    def get_input_arrays(self):
        random_key = jax.random.key(self.random_seed)
        input_arrays = []
        for shape, dtype in self.kernel_wrapper.get_input_structs():
            if jnp.issubdtype(dtype, jnp.floating):
                input_arrays.append(jax.random.uniform(random_key, shape, dtype))
            elif jnp.issubdtype(dtype, jnp.integer):
                input_arrays.append(jax.random.randint(random_key, shape, dtype))
            elif jnp.issubdtype(dtype, jnp.bool_):
                input_arrays.append(jax.random.bernoulli(random_key, shape, dtype))
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
        return input_arrays

    def profile(self):
        self.kernel_wrapper.compile()
        compiled_function = self.kernel_wrapper.get_compiled_function()
        input_arrays = self.get_input_arrays()
        with jax.profiler.trace(self.profile_folder):
            for _ in range(self.iterations):
                compiled_function(*input_arrays).block_until_ready()


    def parse_json_trace(self):
        if not os.path.exists(os.path.join(self.profile_folder, "trace_events.json")):
            trace_parser = TraceParser(self.profile_folder)
            profile_json = trace_parser.read_trace_json()
            if profile_json is None:
                warnings.warn(f"{self.profiler_name}: No trace events found in the data", UserWarning)
                return None
            trace_events = profile_json.get('traceEvents', [])
            if not trace_events:
                warnings.warn(f"{self.profiler_name}: No trace events found in the data", UserWarning)
                return None
            if self.save_to_file:
                # Store the trace events in a file
                with open(os.path.join(self.profile_folder, "trace_events.json"), "w") as f:
                    json.dump(trace_events, f, indent=2)
        else:
            with open(os.path.join(self.profile_folder, "trace_events.json"), "r") as f:
                trace_events = json.load(f)
        self.json_trace_events = trace_events
        return self.json_trace_events


    def filter_trace_events(self):
        def merge_filtered_events_by_name(filtered_events):
            # First group by name
            grouped = {}
            for event in filtered_events:
                event_name = event.get('name', 'unknown')
                if event_name not in grouped:
                    grouped[event_name] = []
                grouped[event_name].append(event)
            
            # Merge events with the same name
            merged_filtered_events = {}
            for event_name, events in grouped.items():
                # Use first event as base
                merged = events[0].copy()
                # Collect dur and ts into lists (only these differ)
                merged['dur'] = [e.get('dur') for e in events if 'dur' in e]
                merged['ts'] = [e.get('ts') for e in events if 'ts' in e]
                # Add repeat count
                merged['repeat_count'] = len(events)
                merged_filtered_events[event_name] = merged
            return merged_filtered_events

        self.filtered_events = []
        for event in self.json_trace_events:
            if "pid" not in event.keys() or event['pid'] != 3:
                continue
            # NOTE: The name of the is "compiled_kernel_function" from sub-function in KernelWarpper.compile()
            if "name" in event.keys() and "compiled_kernel_function" in event['name'] and "args" in event.keys():
                self.filtered_events.append(event)
            elif "args" in event.keys() and "long_name" in event['args'].keys():
                self.filtered_events.append(event)
            else:
                continue

        self.filtered_events = merge_filtered_events_by_name(self.filtered_events)
        if self.save_to_file:
            with open(os.path.join(self.profile_folder, "filtered_events.json"), "w") as f:
                json.dump(self.filtered_events, f, indent=2)
        return self.filtered_events

    def calculate_profiling_statistics(self):
        repeat_count = self.iterations
        total_duration = [0] * repeat_count
        computation_duration = [0] * repeat_count
        memory_duration = [0] * repeat_count
        network_duration = [0] * repeat_count

        for event in self.filtered_events.values():
            if "long_name" not in event['args'].keys():
                continue
            total_duration = list_add(total_duration, event['dur'])
            if "copy" in event['name']:
                memory_duration = list_add(memory_duration, event['dur'])
            else:
                computation_duration = list_add(computation_duration, event['dur'])

        total_stats = calculate_statistics(total_duration)
        computation_stats = calculate_statistics(computation_duration)
        memory_stats = calculate_statistics(memory_duration)
        network_stats = calculate_statistics(network_duration)

        self.profiling_statistics = {
            "total": total_stats,
            "computation": computation_stats,
            "memory": memory_stats,
            "network": network_stats,
            "computation_ratio": computation_stats['mean'] / total_stats['mean'],
            "memory_ratio": memory_stats['mean'] / total_stats['mean'],
            "network_ratio": network_stats['mean'] / total_stats['mean'],
        }
        return self.profiling_statistics

    
    def post_process(self):
        self.parse_json_trace()
        self.filter_trace_events()
        self.calculate_profiling_statistics()

    def profile_and_post_process(self):
        self.profile()
        self.post_process()

    def get_profiling_dataframe_generator(self):
        df_generator = DataFrameGenerator()
        df_generator.add_single_value("operation_name", self.kernel_wrapper.get_kernel_name())
        # add input shapes
        for i, (shape, dtype) in enumerate(self.kernel_wrapper.get_input_structs()):
            for j, dim in enumerate(shape):
                df_generator.add_single_value(f"input_{i}_dim_{j}", dim)
        # add statistics
        df_generator.add_single_value("total_mean", self.profiling_statistics['total']['mean'])
        df_generator.add_single_value("total_std", self.profiling_statistics['total']['std'])
        df_generator.add_single_value("total_min", self.profiling_statistics['total']['min'])
        df_generator.add_single_value("total_max", self.profiling_statistics['total']['max'])
        df_generator.add_single_value("total_median", self.profiling_statistics['total']['median']) 
        df_generator.add_single_value("computation_mean", self.profiling_statistics['computation']['mean'])
        df_generator.add_single_value("computation_std", self.profiling_statistics['computation']['std'])
        df_generator.add_single_value("computation_min", self.profiling_statistics['computation']['min'])
        df_generator.add_single_value("computation_max", self.profiling_statistics['computation']['max'])
        df_generator.add_single_value("computation_median", self.profiling_statistics['computation']['median'])
        df_generator.add_single_value("memory_mean", self.profiling_statistics['memory']['mean'])
        df_generator.add_single_value("memory_std", self.profiling_statistics['memory']['std'])
        df_generator.add_single_value("memory_min", self.profiling_statistics['memory']['min'])
        df_generator.add_single_value("memory_max", self.profiling_statistics['memory']['max'])
        df_generator.add_single_value("memory_median", self.profiling_statistics['memory']['median'])
        df_generator.add_single_value("network_mean", self.profiling_statistics['network']['mean'])
        df_generator.add_single_value("network_std", self.profiling_statistics['network']['std'])
        df_generator.add_single_value("network_min", self.profiling_statistics['network']['min'])
        df_generator.add_single_value("network_max", self.profiling_statistics['network']['max'])
        df_generator.add_single_value("network_median", self.profiling_statistics['network']['median'])
        df_generator.add_single_value("network_duration", self.profiling_statistics['network']['mean'])
        df_generator.add_single_value("computation_utilization", self.profiling_statistics['computation_ratio'])
        df_generator.add_single_value("memory_utilization", self.profiling_statistics['memory_ratio'])
        df_generator.add_single_value("network_utilization", self.profiling_statistics['network_ratio'])
        return df_generator


class KernelProfilerSimpleElementwise(KernelProfilerV0):
    def calculate_profiling_statistics(self):
        repeat_count = self.iterations
        computation_duration = [0] * repeat_count

        for event in self.filtered_events.values():
            if "long_name" not in event['args'].keys():
                continue
            if "copy" not in event['name']:
                computation_duration = list_add(computation_duration, event['dur'])

        computation_mean = sum(computation_duration) / len(computation_duration)
        self.profiling_statistics = {
            "computation_mean": computation_mean,
        }
        return self.profiling_statistics


    def get_profiling_dataframe_generator(self):
        df_generator = DataFrameGenerator()
        df_generator.add_single_value("operation_name", self.kernel_wrapper.get_kernel_name())
        shape = self.kernel_wrapper.get_input_structs()[0][0]
        # Always output 3 dimension columns for consistent merging across 1D/2D/3D shapes
        # Pad shape to 3 dimensions with 1s for unused dimensions
        padded_shape = list(shape) + [1] * (3 - len(shape))
        for i in range(3):
            df_generator.add_single_value(f"input_dim_{i}", padded_shape[i])
        size = np.prod(shape)
        df_generator.add_single_value("size", size)
        df_generator.add_single_value("computation_mean", self.profiling_statistics['computation_mean'])

        return df_generator