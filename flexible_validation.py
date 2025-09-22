import json
import pandas as pd
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Callable, Union
from enum import Enum
from trace_parser import TraceParser
from typing import Optional
import os
from utils import DataFrameGenerator

from kernel_functions import KernelType, ScaleSimTopologyType


# Global variable for trace directory
_TRACE_DIR = "./trace"

def get_trace_dir() -> str:
    """Get the current trace directory."""
    return _TRACE_DIR

def set_trace_dir(trace_dir: str) -> None:
    """Set the trace directory to a new value.
    
    Args:
        trace_dir: New path for the trace directory
    """
    global _TRACE_DIR
    _TRACE_DIR = trace_dir

def setup_trace_dir(trace_dir: Optional[str] = None) -> str:
    if trace_dir is None:
        trace_dir = get_trace_dir()
    else:
        set_trace_dir(trace_dir)
    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)
    return trace_dir


class ValidationConfig:
    def __init__(self, name: str,
        kernel_type: KernelType,
        inputs: List[Tuple[Tuple[int, ...], jnp.dtype]],
        kernel_params: Optional[Dict] = None,
    ):
        """
        Initialize ValidationConfig with flexible inputs and kernel parameters.
        
        Args:
            name: Configuration name
            kernel_type: Type of kernel to use
            inputs: List of (shape, dtype) pairs for each input tensor
            kernel_params: Optional dictionary of additional kernel parameters
        """
        self.name = name
        self.kernel_type = kernel_type
        self.inputs = inputs
        self.kernel_params = kernel_params or {}
        
        
    def get_input_shapes(self) -> List[Tuple[int, ...]]:
        """Get list of input shapes."""
        return [shape for shape, _ in self.inputs]
    
    def get_input_dtypes(self) -> List[jnp.dtype]:
        """Get list of input dtypes."""
        return [dtype for _, dtype in self.inputs]
    
    def get_input_count(self) -> int:
        """Get number of inputs."""
        return len(self.inputs)

    def get_json_dict(self) -> Dict:
        # Convert dtypes to string representation for JSON serialization
        serializable_inputs = []
        for shape, dtype in self.inputs:
            # Extract the dtype name (e.g., jnp.float32 -> "float32")
            dtype_name = dtype.__name__ if hasattr(dtype, '__name__') else str(dtype)
            serializable_inputs.append((shape, dtype_name))
        
        return {
            "name": self.name,
            "kernel_type": self.kernel_type.value,
            "inputs": serializable_inputs,
            "kernel_params": self.kernel_params
        }

    def get_scale_sim_topology_entry(self):
        """
        Return a row (as a list of strings) for the appropriate SCALE-Sim topology CSV config.
        If GEMM, use the matmul config columns: Layer, M, N, K
        If CONV, use the convolution config columns: Layer Name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filters, Strides
        """
        topo_type = self.kernel_type.get_scale_sim_topology_type()
        if topo_type == ScaleSimTopologyType.GEMM:
            # Expecting two input tensors: (M, K) and (K, N)
            # We'll infer M, N, K from the input shapes
            if len(self.inputs) < 2:
                raise ValueError("GEMM config requires at least two input tensors")
            shape_a = self.inputs[0][0]
            shape_b = self.inputs[1][0]
            # Assume A: (M, K), B: (K, N)
            if len(shape_a) != 2 or len(shape_b) != 2:
                raise ValueError("GEMM input shapes must be 2D")
            M, K1 = shape_a
            K2, N = shape_b
            if K1 != K2:
                raise ValueError("GEMM input K dimensions do not match")
            return [self.name, str(M), str(N), str(K1)]
        elif topo_type == ScaleSimTopologyType.CONV:
            # Expecting input tensor and filter tensor
            # Fixed format: input (N, C, H, W), filter (O, I, H, W)
            if len(self.inputs) < 2:
                raise ValueError("CONV config requires at least two input tensors")
            input_shape = self.inputs[0][0]
            filter_shape = self.inputs[1][0]
            
            # Input must be (N, C, H, W)
            if len(input_shape) != 4:
                raise ValueError("CONV input shape must be 4D (N, C, H, W)")
            _, channels, ifmap_h, ifmap_w = input_shape
            
            # Filter must be (O, I, H, W)
            if len(filter_shape) != 4:
                raise ValueError("CONV filter shape must be 4D (O, I, H, W)")
            num_filters, c2, fh, fw = filter_shape
            
            if c2 != channels:
                raise ValueError("CONV input and filter channel count do not match")
            # Stride: default to 1
            stride = 1
            return [self.name, str(ifmap_h), str(ifmap_w), str(fh), str(fw), str(channels), str(num_filters), str(stride)]
        else:
            raise ValueError(f"Unknown topology type: {topo_type}")
    

class ValidationPackage:
    def __init__(self, config: ValidationConfig):
        self.config = config
        # Create ShapeDtypeStruct objects for each input

        
        # Lower the JIT kernel with all input structures
        self.jit_kernel: Optional[Callable] = None # Placeholder for the JIT kernel
        #generate random inputs
        self.inputs = []
        self.output = None # Placeholder for the output

        # Create a profile folder for the validation
        self.profile_dir = get_trace_dir()
        self.profile_folder = os.path.join(self.profile_dir, self.config.name)
        self.profile_json = None
        self.profile_filtered_events: List = []
        if not os.path.exists(self.profile_folder):
            os.makedirs(self.profile_folder)

    def get_profile_folder(self) -> str:
        return self.profile_folder

    def get_json_dict(self) -> Dict:
        return {
            "config": self.config.get_json_dict(),
            "profile_folder": self.profile_folder
        }

    def setup_validation(self):
        input_structs = []
        for shape, dtype in self.config.inputs:
            input_structs.append(jax.ShapeDtypeStruct(shape, dtype))
        
        # Get the kernel function and create a wrapper that includes parameters
        kernel_func = self.config.kernel_type.get_kernel()
        
        if self.config.kernel_params:
            # Create a wrapper function that includes the kernel parameters
            def kernel_with_params(*tensor_inputs):
                return kernel_func(*tensor_inputs, **self.config.kernel_params)
            self.jit_kernel = jax.jit(kernel_with_params).lower(*input_structs).compile()
        else:
            # No parameters, use kernel function directly
            self.jit_kernel = jax.jit(kernel_func).lower(*input_structs).compile()
        
        for shape, dtype in self.config.inputs:
            self.inputs.append(jax.random.normal(jax.random.key(0), shape, dtype))

    def run_validation(self):
        if self.jit_kernel is None:
            raise ValueError("JIT kernel not initialized. Run setup_validation() first.")
        self.output = self.jit_kernel(*self.inputs)
        return self.output

    def profile_validation(self, repeat: int = 1):
        print(f"Profiling {self.config.name}")
        with jax.profiler.trace(self.profile_folder):
            for _ in range(repeat):
                self.run_validation().block_until_ready()
    
    def parse_profile_trace(self):
        if not os.path.exists(os.path.join(self.profile_folder, "trace_events.json")):
            trace_parser = TraceParser(self.profile_folder)
            # trace_parser.parse_trace_csv()
            self.profile_json = trace_parser.read_trace_json()
            if self.profile_json is None:
                print("No trace events found in the data")
                return None
            trace_events = self.profile_json.get('traceEvents', [])
            if not trace_events:
                print("No trace events found in the data")
                return None
            # Store the trace events in a file
            with open(os.path.join(self.profile_folder, "trace_events.json"), "w") as f:
                json.dump(trace_events, f, indent=2)
        else:
            with open(os.path.join(self.profile_folder, "trace_events.json"), "r") as f:
                trace_events = json.load(f)
        self.profile_filtered_events = self.filter_profile_trace_events(trace_events, write_to_file=True)
    
    def filter_profile_trace_events(self, trace_events, write_to_file: bool = False):
        filtered_events = []
        kernel_function_name = self.config.kernel_type.get_kernel().__name__
        for event in trace_events:
            if "pid" not in event.keys() or event['pid'] != 8:
                continue

            if "name" in event.keys() and (kernel_function_name in event['name'] or "jit_kernel" in event['name']) and "args" in event.keys():
                filtered_events.append(event)
            elif "args" in event.keys() and "long_name" in event['args'].keys():
                filtered_events.append(event)
            else:
                continue

        if write_to_file:
            with open(os.path.join(self.profile_folder, "filtered_events.json"), "w") as f:
                json.dump(filtered_events, f, indent=2)
        return filtered_events

    def get_filtered_events_dataframe_generator(self):
        df_generator = DataFrameGenerator()
        for event in self.profile_filtered_events:
            kernel_function_name = self.config.kernel_type.get_kernel().__name__
            if kernel_function_name in event['name'] or "jit_kernel" in event['name']:
                event_type = "main"
            else:
                event_type = "sub"
            df_generator.add_single_value("kernel_name", self.config.name)
            df_generator.add_single_value("event_type", event_type)
            df_generator.add_data("event_name", [event["name"]])
            df_generator.add_data("dur(us)", [event["dur"]])
        return df_generator

    # -----------------------------
    # Additional filtered events data extraction
    def get_filtered_events_dataframe_generator_for_copy_done(self):
        df_generator = DataFrameGenerator()
        for event in self.profile_filtered_events:
            if event["name"] == "copy-done":
                df_generator.add_single_value("kernel_name", self.config.name)
                df_generator.add_data("event_name", [event["name"]])
                df_generator.add_data("dur(us)", [event["dur"]])
                df_generator.add_data("bytes_accessed", [event["args"]["bytes_accessed"]])
                df_generator.add_data("raw_bytes_accessed", [event["args"]["raw_bytes_accessed"]])
                df_generator.add_data("shape_with_layout", [event["args"]["shape_with_layout"]])
        return df_generator

    def get_filtered_events_dataframe_generator_for_avg_fusion_duration(self):
        from collections import defaultdict
        import numpy as np

        # Group durations by event name
        fusion_durations = defaultdict(list)
        for event in self.profile_filtered_events:
            kernel_function_name = self.config.kernel_type.get_kernel().__name__
            if kernel_function_name in event['name'] or "jit_kernel" in event['name'] or "copy" in event['name']:
                continue
            else:
                fusion_durations[event["name"]].append(event["dur"])

        event_df_generator = DataFrameGenerator()
        for event_name, durations in fusion_durations.items():
            avg_duration = float(np.mean(durations))
            event_df_generator.add_single_value("kernel_name", self.config.name)
            event_df_generator.add_data("event_name", [event_name])
            event_df_generator.add_data("dur(us)", [avg_duration])

        # Sum all event durations, one row df with two columns: kernel_name and dur
        import numpy as np
        durations = event_df_generator.data.get("dur(us)", [])
        total_duration = float(np.sum(durations)) if durations else 0.0
        df_generator = DataFrameGenerator()
        df_generator.add_single_value("kernel_name", self.config.name)
        df_generator.add_single_value("dur(us)", total_duration)
        return df_generator

    # -----------------------------

    def get_output(self) -> jnp.ndarray:
        if self.output is None:
            raise ValueError("Output not available. Run run_validation() first.")
        return self.output


class ValidationManager:
    def __init__(self, profile_dir: Optional[str] = None):
        self.packages: List[ValidationPackage] = []
        setup_trace_dir(profile_dir)
        self.profile_dir = get_trace_dir()
        print(f"Profile directory: {self.profile_dir}")

    def add_config(self, config: ValidationConfig):
        self.packages.append(ValidationPackage(config))

    def clear_packages(self):
        self.packages = []

    def profile_all_packages(self, repeat: int = 1):
        for package in self.packages:
            package.setup_validation()
            package.profile_validation(repeat = repeat)
            # package.parse_profile_trace()

    def parse_all_packages(self):
        for package in self.packages:
            package.parse_profile_trace()

    def save_package_to_json(self, book_keeping_file: str):
        with open(book_keeping_file, "w") as f:
            json.dump([package.get_json_dict() for package in self.packages], f, indent=2)

    def load_package_from_json(self, book_keeping_file: str):
        with open(book_keeping_file, "r") as f:
            packages = json.load(f)
            self.packages = []
            for package in packages:
                # Convert kernel_type string back to KernelType
                kernel_type = KernelType(package["config"]["kernel_type"])
                # Convert input shapes from lists to tuples, and dtypes from string to jnp.dtype
                inputs = []
                for shape, dtype in package["config"]["inputs"]:
                    if isinstance(dtype, str):
                        # Parse dtype string (e.g., "float32" -> jnp.float32)
                        try:
                            dtype_obj = getattr(jnp, dtype)
                        except AttributeError:
                            # Fallback for common dtypes
                            dtype_mapping = {
                                'float32': jnp.float32,
                                'float64': jnp.float64,
                                'int32': jnp.int32,
                                'int64': jnp.int64,
                                'bool': jnp.bool_,
                                'complex64': jnp.complex64,
                                'complex128': jnp.complex128
                            }
                            dtype_obj = dtype_mapping.get(dtype, jnp.float32)
                    else:
                        dtype_obj = dtype
                    inputs.append((tuple(shape), dtype_obj))
                # Get kernel parameters if they exist
                kernel_params = package["config"].get("kernel_params", {})
                
                config = ValidationConfig(
                    name=package["config"]["name"],
                    kernel_type=kernel_type,
                    inputs=inputs,
                    kernel_params=kernel_params
                )
                self.packages.append(ValidationPackage(config))

    def get_filtered_events_dataframe(self, save_to_file: bool = True):
        df_generator = DataFrameGenerator()
        for package in self.packages:
            df_generator.merge(package.get_filtered_events_dataframe_generator())
        if save_to_file:
            df_generator.to_dataframe().to_csv(os.path.join(self.profile_dir, "filtered_events.csv"), index=False)
        return df_generator.to_dataframe()

    # -----------------------------
    # Custom data extraction

    def get_filtered_events_dataframe_for_copy_done(self, save_to_file: bool = True):
        df_generator = DataFrameGenerator()
        for package in self.packages:
            df_generator.merge(package.get_filtered_events_dataframe_generator_for_copy_done())
        if save_to_file:
            df_generator.to_dataframe().to_csv(os.path.join(self.profile_dir, "filtered_events_copy_done.csv"), index=False)
        return df_generator.to_dataframe()

    def get_filtered_events_dataframe_for_avg_fusion_duration(self, save_to_file: bool = True):
        df_generator = DataFrameGenerator()
        for package in self.packages:
            df_generator.merge(package.get_filtered_events_dataframe_generator_for_avg_fusion_duration())
        if save_to_file:
            df_generator.to_dataframe().to_csv(os.path.join(self.profile_dir, "filtered_events_avg_fusion.csv"), index=False)
        return df_generator.to_dataframe()

    # -----------------------------


    def write_scale_sim_topology_csv(self):
        """Write SCALE-Sim topology CSV files, separating GEMM and CONV configurations."""
        gemm_entries = []
        conv_entries = []
        
        for package in self.packages:
            topo_entry = package.config.get_scale_sim_topology_entry()
            topo_type = package.config.kernel_type.get_scale_sim_topology_type()
            
            if topo_type == ScaleSimTopologyType.GEMM:
                gemm_entries.append(topo_entry)
            elif topo_type == ScaleSimTopologyType.CONV:
                conv_entries.append(topo_entry)
        
        # Write GEMM topology file
        if gemm_entries:
            gemm_df = pd.DataFrame(gemm_entries)
            gemm_df.columns = ["Layer", "M", "N", "K"]
            # Add empty column to ensure each line ends with comma
            gemm_df[""] = ""
            gemm_file = os.path.join(self.profile_dir, "scale_sim_gemm_topology.csv")
            gemm_df.to_csv(gemm_file, index=False)
            print(f"GEMM topology written to: {gemm_file}")
        
        # Write CONV topology file
        if conv_entries:
            conv_df = pd.DataFrame(conv_entries)
            conv_df.columns = [
                "Layer Name", "IFMAP Height", "IFMAP Width", 
                "Filter Height", "Filter Width", "Channels", "Num Filters", "Strides"
            ]
            # Add empty column to ensure each line ends with comma
            conv_df[""] = ""
            conv_file = os.path.join(self.profile_dir, "scale_sim_conv_topology.csv")
            conv_df.to_csv(conv_file, index=False)
            print(f"CONV topology written to: {conv_file}")
        
        if not gemm_entries and not conv_entries:
            print("No topology entries found to write.")

