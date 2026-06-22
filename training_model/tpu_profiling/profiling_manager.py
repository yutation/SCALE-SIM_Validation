from ast import List
import csv
import os
from typing import Any, Dict, Optional

import jax
import kernel_profiler as kp
import jax_kernel_functions as jkf
from utils import DataFrameGenerator
from datetime import datetime
from profiling_configuration import CODE_VERSION



class ProfilingManagerBase:
    def __init__(self, manager_name: str, manager_dir: str, configuration: Optional[Dict[str, Any]] = None):
        self.manager_name = manager_name
        self.manager_dir = manager_dir
        self.profile_dir = os.path.join(self.manager_dir, self.manager_name)
        self.profilers: List[kp.KernelProfilerBase] = []
        self.configuration: Dict[str, Any] = configuration
        self.profiler_name_list: List[str] = []

    def add_profiler(self, profiler_name: str, kernel_wrapper: jkf.KernelWarpper, configuration: Optional[Dict[str, Any]] = None):
        pass

    def profile_all_profilers(self):
        for profiler in self.profilers:
            profiler.profile()

    def post_process_all_profilers(self):
        for profiler in self.profilers:
            profiler.post_process()

    def profile_and_post_process_all_profilers(self):
        total_profilers = len(self.profilers)
        checkpoint_interval = 20
        
        print("\n" + "=" * 80)
        print(f"[PROFILING PROGRESS] Starting: {total_profilers} profilers")
        print("=" * 80 + "\n")
        
        for idx, profiler in enumerate(self.profilers, 1):
            profiler.profile_and_post_process()
            
            # Print every 100 profilers
            if idx % checkpoint_interval == 0:
                print("\n" + "=" * 80)
                print(f"[PROFILING PROGRESS] Completed: {idx}/{total_profilers} profilers ({idx/total_profilers*100:.1f}%)")
                print("=" * 80 + "\n")
        
        print("\n" + "=" * 80)
        print(f"[PROFILING PROGRESS] Complete: {total_profilers}/{total_profilers} profilers (100.0%)")
        print("=" * 80 + "\n")

    def get_profiling_dataframe_generator_all_profilers(self):
        df_generator = DataFrameGenerator()
        for profiler in self.profilers:
            df_generator.merge(profiler.get_profiling_dataframe_generator())
        return df_generator


class ProfilingManagerV0(ProfilingManagerBase):
    def __init__(self, manager_name: str, manager_dir: str, configuration: Optional[Dict[str, Any]] = None):
        super().__init__(manager_name, manager_dir, configuration)
        
        self.storage_metadata_file = configuration.get("storage_metadata_file", None)
        self.append_to_metadata_file = configuration.get("append_to_metadata_file", True)
        self.storage_file = configuration.get("storage_file", None)

        self.matedata_dfg = DataFrameGenerator()
        self.matedata_dfg.add_single_value("storage_file", configuration.get("storage_file", None))
        self.hardware_config = configuration.get("hardware_config", "TPUv4")
        self.matedata_dfg.add_single_value("operator_name", configuration.get("operator_name", "unknown"))
        #TODO Why need this?
        self.matedata_dfg.add_single_value("common_operator_dimensions", configuration.get("common_operator_dimensions", None))
        self.matedata_dfg.add_single_value("data_precision", configuration.get("data_precision", "FP16"))
        self.matedata_dfg.add_single_value("iterations", configuration.get("profiler_iterations", 10))
        self.matedata_dfg.add_single_value("random_seed", configuration.get("random_seed", 0))
        self.matedata_dfg.add_single_value("software_version", f"jax={jax.__version__}")
        self.matedata_dfg.add_single_value("date", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.matedata_dfg.add_single_value("script_version", CODE_VERSION)
        self.matedata_dfg.add_single_value("repo_version", configuration.get("repo_version", "unknown"))
        self.matedata_dfg.add_single_value("comment", configuration.get("comment", ""))

        if not os.path.exists(self.storage_metadata_file) or not self.append_to_metadata_file:
            os.makedirs(os.path.dirname(self.storage_metadata_file), exist_ok=True)
            with open(self.storage_metadata_file, "w") as f:
                header: List[str] = self.matedata_dfg.get_header()
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()


        self.profiler_configuration: Dict[str, Any] = {
            "random_seed": 0,
            "iterations": 10,
            "save_to_file": True,
        }

    def add_profiler(self, profiler_name: str, kernel_wrapper: jkf.KernelWarpper):
        if profiler_name in self.profiler_name_list:
            raise ValueError(f"Profiler name {profiler_name} already exists")
        self.profiler_name_list.append(profiler_name)
        profiler = kp.KernelProfilerV0(profiler_name, kernel_wrapper, self.profile_dir, self.profiler_configuration)
        self.profilers.append(profiler)

    def write_results(self):
        storage_dataframe_generator = self.get_profiling_dataframe_generator_all_profilers()
        storage_dataframe_generator.to_dataframe().to_csv(self.storage_file, index=False)
        if self.storage_metadata_file is not None:
            with open(self.storage_metadata_file, "a") as f:
                row = self.matedata_dfg.get_row_dict(0)
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)

class ProfilingManagerSimpleElementwise(ProfilingManagerV0):
    def __init__(self, manager_name: str, manager_dir: str, configuration: Optional[Dict[str, Any]] = None):
        super().__init__(manager_name, manager_dir, configuration)

    def add_profiler(self, profiler_name: str, kernel_wrapper: jkf.KernelWarpper):
        if profiler_name in self.profiler_name_list:
            raise ValueError(f"Profiler name {profiler_name} already exists")
        self.profiler_name_list.append(profiler_name)
        profiler = kp.KernelProfilerSimpleElementwise(profiler_name, kernel_wrapper, self.profile_dir, self.profiler_configuration)
        self.profilers.append(profiler)